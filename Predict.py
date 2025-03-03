import numpy as np
import pandas as pd
import math
import time

import psycopg2
from sqlalchemy import create_engine

from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix, issparse

class OptimizedRecommenderSimilarity:
    def __init__(self, 
                 dbname: str, 
                 user: str, 
                 password: str, 
                 host: str = 'localhost', 
                 port: int = 5432):
        """
        Initialize database connection and matrix optimization tools
        """
        # Database connection
        self.connection_string = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
        self.engine = create_engine(self.connection_string)
    
    def _fetch_ratings_matrix(self, 
                               min_ratings: int = 0, 
                               max_users: int = math.inf,
                               max_dishes: int = math.inf):
        """
        Fetch ratings matrix with optimization strategies
        
        Args:
        - min_ratings: Minimum number of ratings to include a user
        - max_users: Maximum number of users to process
        - max_dishes: Maximum number of dishes to include
        
        Returns:
        - Sparse ratings matrix
        - User and dish mappings
        """
        # Efficient query to get relevant ratings
        query = f"""
        WITH user_rating_counts AS (
            SELECT 
                user_id, 
                COUNT(DISTINCT dish_id) as dish_count
            FROM ratings
            GROUP BY user_id
            HAVING COUNT(DISTINCT dish_id) >= {min_ratings}
            ORDER BY dish_count DESC
            LIMIT {max_users}
        ),
        top_dishes AS (
            SELECT dish_id
            FROM (
                SELECT 
                    dish_id, 
                    COUNT(DISTINCT user_id) as user_count
                FROM ratings
                GROUP BY dish_id
                ORDER BY user_count DESC
                LIMIT {max_dishes}
            ) top_dish_subquery
        )
        SELECT 
            rc.user_id, 
            r.dish_id, 
            r.rating
        FROM ratings r
        JOIN user_rating_counts rc ON r.user_id = rc.user_id
        JOIN top_dishes td ON r.dish_id = td.dish_id
        ORDER BY rc.user_id, r.dish_id
        """
        
        # Fetch ratings efficiently
        ratings_df = pd.read_sql(query, self.engine)
        
        # Create mappings
        user_mapping = {user: idx for idx, user in enumerate(ratings_df['user_id'].unique())}
        dish_mapping = {dish: idx for idx, dish in enumerate(ratings_df['dish_id'].unique())}
        
        # Map users and dishes to matrix indices
        ratings_df['user_idx'] = ratings_df['user_id'].map(user_mapping)
        ratings_df['dish_idx'] = ratings_df['dish_id'].map(dish_mapping)
        
        # Create sparse matrix
        sparse_matrix = csr_matrix(
            (ratings_df['rating'], 
             (ratings_df['user_idx'], ratings_df['dish_idx'])),
            shape=(len(user_mapping), len(dish_mapping))
        )
        
        return sparse_matrix, user_mapping, dish_mapping
    
    def compute_similarity_optimized(self, 
                                     normalization: str ='standard', 
                                     similarity_threshold: float = 0.5,
                                     max_neighbors: int = math.inf):
        """
        Compute user similarities with multiple optimizations
        
        Args:
        - normalization: Normalization method ('standard' or 'min-max')
        - similarity_threshold: Minimum similarity to keep
        - max_neighbors: Maximum similar users to compute
        
        Returns:
        - Similarity matrix
        - User mapping
        """
        start_time = time.time()
        
        # Fetch optimized ratings matrix
        ratings_matrix, user_mapping, _dish_mapping = self._fetch_ratings_matrix()
        
        # Convert to dense for similarity (if not already)
        if issparse(ratings_matrix):
            ratings_array = ratings_matrix.toarray()
        else:
            ratings_array = ratings_matrix
        
        # Normalization
        if normalization == 'standard':
            scaler = StandardScaler()
            normalized_ratings = scaler.fit_transform(ratings_array)
        else:
            # Min-Max scaling alternative
            normalized_ratings = (ratings_array - ratings_array.min()) / (ratings_array.max() - ratings_array.min())
        
        # Compute cosine similarity with dimensionality reduction
        from sklearn.decomposition import TruncatedSVD
        
        # Dimensionality reduction before similarity
        svd = TruncatedSVD(n_components=min(50, normalized_ratings.shape[1]), random_state=42)
        reduced_ratings = svd.fit_transform(normalized_ratings)
        
        # Compute cosine similarity on reduced dimensions
        similarity_matrix = cosine_similarity(reduced_ratings)
        
        # Threshold and sparsify similarity matrix
        similarity_matrix[similarity_matrix < similarity_threshold] = 0
        
        # Sort and keep only top N neighbors
        for i in range(len(similarity_matrix)):
            # Get indices of similar users, sorted by similarity
            similar_indices = np.argsort(similarity_matrix[i])[::-1]
            
            # Keep only top max_neighbors
            mask = np.zeros_like(similarity_matrix[i], dtype=bool)
            mask[similar_indices[:max_neighbors]] = True
            
            # Zero out others
            similarity_matrix[i][~mask] = 0
        
        end_time = time.time()
        print(f"Similarity Computation Time: {end_time - start_time:.2f} seconds")
        
        return similarity_matrix, user_mapping
    
    def store_similarities(self, similarity_matrix, user_mapping):
        """
        Store computed similarities back to database
        
        Args:
        - similarity_matrix: Computed similarity matrix
        - user_mapping: Mapping of users to matrix indices
        """
        # Reverse user mapping
        reverse_user_mapping = {idx: user for user, idx in user_mapping.items()}
        
        # Prepare similarity data for insertion
        similarities_data = []
        for i in range(len(similarity_matrix)):
            for j in range(len(similarity_matrix)):
                if i != j and similarity_matrix[i, j] > 0:
                    similarities_data.append({
                        'source_user_id': reverse_user_mapping[i],
                        'target_user_id': reverse_user_mapping[j],
                        'similarity_score': float(similarity_matrix[i, j])
                    })
        
        # Convert to DataFrame and bulk insert
        similarities_df = pd.DataFrame(similarities_data)
        
        # Bulk insert with SQLAlchemy
        try:
            similarities_df.to_sql(
                'user_similarities', 
                self.engine, 
                if_exists='replace',  # Replace existing table
                index=False,
                method='multi'  # Faster insertion
            )
        except Exception as e:
            print(f"Error storing similarities: {e}")

class ContentBasedRecommender:
    def __init__(self, 
                 dbname: str, 
                 user: str, 
                 password: str, 
                 host: str = 'localhost', 
                 port: int = 5432):
        """
        Initialize database connection and matrix optimization tools
        """
        # Database connection
        self.connection_string = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
        self.engine = create_engine(self.connection_string)

    def _fetch_content_matrix(self, 
                            min_content: int = 0, 
                            max_restaurants: int = math.inf,
                            max_dishes: int = math.inf):
        """
        Fetch content matrix with optimization strategies
        
        Args:
        - min_ratings: Minimum number of dishes to include for a restaraunt
        - max_users: Maximum number of restaraunts to process
        - max_dishes: Maximum number of dishes to include
        
        Returns:
        - Sparse ratings matrix
        - User and dish mappings
        """
        # Efficient query to get relevant ratings
        query = f"""
        WITH user_rating_counts AS (
            SELECT 
                user_id, 
                COUNT(DISTINCT dish_id) as dish_count
            FROM ratings
            GROUP BY user_id
            HAVING COUNT(DISTINCT dish_id) >= {min_content}
            ORDER BY dish_count DESC
            LIMIT {max_restaurants}
        ),
        top_dishes AS (
            SELECT dish_id
            FROM (
                SELECT 
                    dish_id, 
                    COUNT(DISTINCT user_id) as user_count
                FROM ratings
                GROUP BY dish_id
                ORDER BY user_count DESC
                LIMIT {max_dishes}
            ) top_dish_subquery
        )
        SELECT 
            rc.user_id, 
            r.dish_id, 
            r.rating
        FROM ratings r
        JOIN user_rating_counts rc ON r.user_id = rc.user_id
        JOIN top_dishes td ON r.dish_id = td.dish_id
        ORDER BY rc.user_id, r.dish_id
        """
        
        # Fetch ratings efficiently
        ratings_df = pd.read_sql(query, self.engine)
        
        # Create mappings
        restaurant_mapping = {restaurant: idx for idx, restaraunt in enumerate(ratings_df['user_id'].unique())}
        dish_mapping = {dish: idx for idx, dish in enumerate(ratings_df['dish_id'].unique())}
        
        # Map users and dishes to matrix indices
        ratings_df['restaurant_idx'] = ratings_df['restaurant_id'].map(restaurant_mapping)
        ratings_df['dish_idx'] = ratings_df['dish_id'].map(dish_mapping)
        
        # Create sparse matrix
        sparse_matrix = csr_matrix(
            (ratings_df['rating'], 
                (ratings_df['user_idx'], ratings_df['dish_idx'])),
            shape=(len(user_mapping), len(dish_mapping))
        )
        
        return sparse_matrix, user_mapping, dish_mapping
    
    def compute_similarity_optimized(self, 
                                     normalization: str ='standard', 
                                     similarity_threshold: float = 0.5,
                                     max_neighbors: int = math.inf):
        """
        Compute user similarities with multiple optimizations
        
        Args:
        - normalization: Normalization method ('standard' or 'min-max')
        - similarity_threshold: Minimum similarity to keep
        - max_neighbors: Maximum similar users to compute
        
        Returns:
        - Similarity matrix
        - User mapping
        """
        start_time = time.time()
        
        # Fetch optimized ratings matrix
        ratings_matrix, user_mapping, _dish_mapping = self._fetch_ratings_matrix()
        
        # Convert to dense for similarity (if not already)
        if issparse(ratings_matrix):
            ratings_array = ratings_matrix.toarray()
        else:
            ratings_array = ratings_matrix
        
        # Normalization
        if normalization == 'standard':
            scaler = StandardScaler()
            normalized_ratings = scaler.fit_transform(ratings_array)
        else:
            # Min-Max scaling alternative
            normalized_ratings = (ratings_array - ratings_array.min()) / (ratings_array.max() - ratings_array.min())
        
        # Compute cosine similarity with dimensionality reduction
        from sklearn.decomposition import TruncatedSVD
        
        # Dimensionality reduction before similarity
        svd = TruncatedSVD(n_components=min(50, normalized_ratings.shape[1]), random_state=42)
        reduced_ratings = svd.fit_transform(normalized_ratings)
        
        # Compute cosine similarity on reduced dimensions
        similarity_matrix = cosine_similarity(reduced_ratings)
        
        # Threshold and sparsify similarity matrix
        similarity_matrix[similarity_matrix < similarity_threshold] = 0
        
        # Sort and keep only top N neighbors
        for i in range(len(similarity_matrix)):
            # Get indices of similar users, sorted by similarity
            similar_indices = np.argsort(similarity_matrix[i])[::-1]
            
            # Keep only top max_neighbors
            mask = np.zeros_like(similarity_matrix[i], dtype=bool)
            mask[similar_indices[:max_neighbors]] = True
            
            # Zero out others
            similarity_matrix[i][~mask] = 0
        
        end_time = time.time()
        print(f"Similarity Computation Time: {end_time - start_time:.2f} seconds")
        
        return similarity_matrix, user_mapping
    
    def store_similarities(self, similarity_matrix, user_mapping):
        """
        Store computed similarities back to database
        
        Args:
        - similarity_matrix: Computed similarity matrix
        - user_mapping: Mapping of users to matrix indices
        """
        # Reverse user mapping
        reverse_user_mapping = {idx: user for user, idx in user_mapping.items()}
        
        # Prepare similarity data for insertion
        similarities_data = []
        for i in range(len(similarity_matrix)):
            for j in range(len(similarity_matrix)):
                if i != j and similarity_matrix[i, j] > 0:
                    similarities_data.append({
                        'source_user_id': reverse_user_mapping[i],
                        'target_user_id': reverse_user_mapping[j],
                        'similarity_score': float(similarity_matrix[i, j])
                    })
        
        # Convert to DataFrame and bulk insert
        similarities_df = pd.DataFrame(similarities_data)
        
        # Bulk insert with SQLAlchemy
        try:
            similarities_df.to_sql(
                'dish_similarities', 
                self.engine, 
                if_exists='replace',  # Replace existing table
                index=False,
                method='multi'  # Faster insertion
            )
        except Exception as e:
            print(f"Error storing similarities: {e}")
    
class MatrixFactorizationRecommender:
    def __init__(self, 
                 dbname: str, 
                 user: str, 
                 password: str, 
                 host: str = 'localhost', 
                 port: int = 5432):
        """
        Initialize database connection and matrix optimization tools
        """
        # Database connection
        self.connection_string = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
        self.engine = create_engine(self.connection_string)
    

class HybridRecommender:
    def __init__(self, dbname, user, password, host='localhost', port=5432):
        self.connection_string = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
        self.engine = create_engine(self.connection_string)
        self.cosine_recommender = OptimizedRecommenderSimilarity(dbname, user, password, host, port)
        self.svd_recommender = MatrixFactorizationRecommender(dbname, user, password, host, port)
        self.content_recommender = ContentBasedRecommender(dbname, user, password, host, port)

    def hybrid_recommendation(self, user_id, alpha=0.5, beta=0.3, gamma=0.2):
        """
        Combine recommendations from different models using weighted averaging.
        """
        # Get recommendations from each model
        cosine_recs = self.cosine_recommender.recommend_dishes(user_id)
        svd_recs = self.svd_recommender.recommend_dishes_svd(user_id)
        content_recs = self.content_recommender.recommend_dishes_content(user_id)

        # Aggregate recommendations
        final_recommendations = []
        for dish in cosine_recs:
            # Weighting the different model recommendations
            weighted_recommendation = alpha * cosine_recs[dish] + beta * svd_recs[dish] + gamma * content_recs[dish]
            final_recommendations.append(weighted_recommendation)

        return final_recommendations
    
# Example usage
def main():
    # Initialize recommender
    recommender = HybridRecommender(
        dbname='your_database',
        user='your_username',
        password='your_password'
    )
    
    # Compute optimized similarities
    similarity_matrix, user_mapping = recommender.compute_similarity_optimized()
    
    # Optionally store back to database
    recommender.store_similarities(similarity_matrix, user_mapping)

if __name__ == "__main__":
    main()