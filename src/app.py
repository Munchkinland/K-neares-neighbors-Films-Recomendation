from utils import db_connect
engine = db_connect()

# your code here
Step 1. Data Ingestion

import pandas as pd

movies = pd.read_csv("https://raw.githubusercontent.com/4GeeksAcademy/k-nearest-neighbors-project-tutorial/main/tmdb_5000_movies.csv")
credits = pd.read_csv("https://raw.githubusercontent.com/4GeeksAcademy/k-nearest-neighbors-project-tutorial/main/tmdb_5000_credits.csv")

movies.head()

credits.head()

movies.to_csv(r"movies.csv", index=False)

credits.to_csv(r"credits.csv", index=False)

"""Step 2. Creating database

2.1 Creating movies and credits tables
"""

# Paths of uploaded files
path_movies = '/workspace/K-nearest-neighbors-films-recomendation/data/raw/credits.csv'
path_credits = '/workspace/K-nearest-neighbors-films-recomendation/data/raw/credits.csv'

# Loading data into Pandas DataFrames
df_movies_uploaded = pd.read_csv(path_movies)
df_credits_uploaded = pd.read_csv(path_credits)

# Showing tables headers
df_movies_uploaded_head = df_movies_uploaded.head()
df_credits_uploaded_head = df_credits_uploaded.head()

df_movies_uploaded_head, df_credits_uploaded_head

df_credits_uploaded.info

df_movies_uploaded.columns

"""Merging using SQL to create a third table. I will use the title of the movie as a key to the connection.

The union of the "movies" and "credits" tables has been carried out successfully, creating a new DataFrame that contains the requested columns: movie_id, title, overview, genres, keywords, cast, and crew.
"""

import sqlite3

conn = sqlite3.connect("../data/movies_database.db")

movies.to_sql("movies_table", conn, if_exists = "replace", index = False)
credits.to_sql("credits_table", conn, if_exists = "replace", index = False)

# Merge tables for creating a new DataFrame

query = """
    SELECT *
    FROM movies_table
    INNER JOIN credits_table
    ON movies_table.title = credits_table.title;
"""

total_data = pd.read_sql_query(query, conn)
conn.close()

total_data = total_data.loc[:, ~total_data.columns.duplicated()]
total_data.head()

"""Step 3. Data transformation"""

# Functions for data transformation
import json
def extract_names_from_json(json_str):
    return [item['name'] for item in json.loads(json_str)]

def extract_cast_names(json_str):
    cast_list = json.loads(json_str)
    return [cast['name'] for cast in cast_list[:3]]  # Primeros tres actores

def extract_director_name(json_str):
    crew_list = json.loads(json_str)
    for crew_member in crew_list:
        if crew_member['job'] == 'Director':
            return crew_member['name']
    return None

# Transforming the 'genres' and 'keywords' columns
total_data['genres'] = total_data['genres'].apply(extract_names_from_json)
total_data['keywords'] = total_data['keywords'].apply(extract_names_from_json)

# Transforming the 'cast' column
total_data['cast'] = total_data['cast'].apply(extract_cast_names)

# Transforming 'crew' column to get just the director name
total_data['crew'] = total_data['crew'].apply(extract_director_name)

# Converting 'overview' column to a list of words
total_data['overview'] = total_data['overview'].apply(lambda x: x.split() if isinstance(x, str) else [])

# Show changes
print(total_data[['genres', 'keywords', 'cast', 'crew', 'overview']].head())

def remove_spaces(items):
    if isinstance(items, list):
        return [item.replace(" ", "") for item in items]
    return items

total_data['genres'] = total_data['genres'].apply(remove_spaces)
total_data['cast'] = total_data['cast'].apply(remove_spaces)
total_data['crew'] = total_data['crew'].apply(lambda x: x.replace(" ", "") if x else x)
total_data['keywords'] = total_data['keywords'].apply(remove_spaces)

# Mostrar los cambios
print(total_data[['genres', 'cast', 'crew', 'keywords']].head())

def clean_and_join(lst):
    if lst is None:
        return ''
    return ' '.join(str(item) for item in lst).replace('[', '').replace(']', '').replace("'", '').replace(',', '')

# Applying the transformation to the columns and combining them into 'tags'
total_data['tags'] = total_data['overview'].apply(clean_and_join) + ' ' + \
                     total_data['genres'].apply(clean_and_join) + ' ' + \
                     total_data['keywords'].apply(clean_and_join) + ' ' + \
                     total_data['cast'].apply(clean_and_join) + ' ' + \
                     total_data['crew'].apply(clean_and_join)

# Show the first element of the 'tags' column to verify
print(total_data['tags'].iloc[0])

"""Step 4. Creating a KNN model

4.1 Text Vectorizing
"""

from sklearn.feature_extraction.text import TfidfVectorizer

# Creating the TF-IDF vectorizer
tfidf = TfidfVectorizer(stop_words='english')

# Vectorizing the 'tags' column
tfidf_matrix = tfidf.fit_transform(total_data['tags'])

"""4.2 Aplying KNN Model"""

from sklearn.neighbors import NearestNeighbors

# Creating the KNN model
model = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='cosine')
model.fit(tfidf_matrix)

import pickle

# Save the KNN model to a file
with open('/workspace/K-nearest-neighbors-films-recomendation/models/knn_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

# Now the KNN model has been saved in the file knn_model.pkl

"""Movie List Consulting"""

show_movie_list = lambda: print("List of movies in the database:\n- " + "\n- ".join(total_data["title"]))
show_movie_list()

# Optimized get_movie_recommendations function
def get_movie_recommendations(tfidf_matrix, model, user_input, total_data):
    movie_indices = total_data.index[total_data["title"] == user_input].tolist()

    if not movie_indices:
        print(f"Sorry, the movie '{user_input}' is not found in the database.")
        return []

    movie_index = movie_indices[0]
    distances, indices = model.kneighbors(tfidf_matrix[movie_index])

    similar_movies = [(total_data["title"][i], distances[0][j]) for j, i in enumerate(indices[0])]
    similar_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)

    recommended_movies = [movie[0] for movie in similar_movies if movie[0] != user_input]

    return recommended_movies

# Main function for user interaction
def main():
    while True:
        user_input = input("Enter the title of the movie to get recommendations (or 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break
        recommendations = get_movie_recommendations(tfidf_matrix, model, user_input, total_data)
        if recommendations:
            print("\nMovie recommendations for '{}':".format(user_input))
            for movie in recommendations:
                print("- Movie: {}".format(movie))

if __name__ == "__main__":
    main()