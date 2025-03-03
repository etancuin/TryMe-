import pandas as pd

import psycopg2

from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

class RestaurantClustering:
    def __init__(self, dbname, user, password, host='localhost', port=5432):
        self.connection_params = {
            "dbname": dbname,
            "user": user,
            "password": password,
            "host": host,
            "port": port
        }
        self.connection = None

    def connect(self):
        if self.connection is None or self.connection.closed:
            self.connection = psycopg2.connect(**self.connection_params)

    def fetch_restaurant_data(self):
        """
        Fetch data about restaurants from the database (features like location, cuisine type, rating, etc.)
        """
        self.connect()
        query = """
        SELECT restaurant_id, cuisine_type, location, avg_rating
        FROM restaurants
        """
        restaurant_df = pd.read_sql_query(query, self.connection)
        return restaurant_df
    
    def cluster_restaurants(self, n_clusters=5):
        """
        Cluster restaurants by their features (cuisine type, location, average rating)
        """
        restaurant_df = self.fetch_restaurant_data()

        # Preprocessing and scaling features
        features = restaurant_df[['avg_rating']]  # Can add other features like location, cuisine type
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)

        # Clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        restaurant_df['cluster'] = kmeans.fit_predict(scaled_features)

        return restaurant_df

class DishClassifier:
    def __init__(self, dbname, user, password, host='localhost', port=5432):
        self.connection_params = {
            "dbname": dbname,
            "user": user,
            "password": password,
            "host": host,
            "port": port
        }
        self.connection = None

    def connect(self):
        if self.connection is None or self.connection.closed:
            self.connection = psycopg2.connect(**self.connection_params)

    def fetch_dish_data(self):
        """
        Fetch data about dishes from the database (features like ingredients, dish type, etc.)
        """
        self.connect()
        query = """
        SELECT dish_id, restaurant_id, ingredients, dish_type
        FROM dishes
        """
        dishes_df = pd.read_sql_query(query, self.connection)
        return dishes_df

    def train_dish_classifier(self):
        """
        Train a classifier to predict dish categories (appetizers, main course, desserts, etc.)
        """
        dishes_df = self.fetch_dish_data()

        # Preprocessing (you'll need to transform features into a usable format)
        X = dishes_df[['ingredients']]  # Feature engineering on 'ingredients' or others
        y = dishes_df['dish_type']  # Target variable

        # Convert text data (ingredients) to numeric representation using CountVectorizer or TF-IDF
        vectorizer = TfidfVectorizer()
        X_tfidf = vectorizer.fit_transform(X['ingredients'])

        # Train Random Forest classifier
        clf = RandomForestClassifier()
        clf.fit(X_tfidf, y)

        return clf, vectorizer
