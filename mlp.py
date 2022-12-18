import numpy as np
import pandas as pd
from tensorflow import keras
import tensorflow as tf
from keras.utils import plot_model
from sklearn.model_selection import train_test_split


df = pd.read_csv('rating.csv')
df = df.iloc[:200000,:]

X = df[['userId','movieId']]
y = df['rating']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
users = df.userId.unique().tolist()
movies = df.movieId.unique().tolist()

movies_df = pd.read_csv('movie.csv')

movie2movie_encoded = {x: i for i,x in enumerate(movies)}
movie_encoded2movie = {i: x for i,x in enumerate(movies)}
user2user_encoded = {x: i for i, x in enumerate(users)}
user_encoded2user = {i: x for i, x in enumerate(users)}

def load_model():
    model = keras.models.load_model('model.h5')
    return model

def get_top_ratings(ratings, k=10):
    top_ratings = ratings.argsort()[-k:]
    top_ratings = [movie_encoded2movie.get([x][0]) for x in top_ratings]
    return top_ratings

def predict(user_id):
    user_id = np.uint64(user_id)
    watched_movs = df[df.userId == user_id].iloc[:,1]
    not_watched_movs = movies_df[~movies_df.movieId.isin(watched_movs.values)].movieId

    not_watched_movs = list(
        set(not_watched_movs).intersection(set(movie2movie_encoded.keys()))
    )
    not_watched_movs = [[movie2movie_encoded.get(x)] for x in not_watched_movs]

    user_encoder = user2user_encoded.get(user_id)
    user_movie_array = np.hstack(([[user_encoder]] * len(not_watched_movs), not_watched_movs))

    model = load_model()

    ratings = model.predict([user_movie_array[:,0],user_movie_array[:,1]]).flatten()
    top_ratings = get_top_ratings(ratings,10)
    top_movies = movies_df[movies_df.movieId.isin(top_ratings)]
    movie_titles = []
    for element in top_movies.itertuples():
        movie_titles.append(element.title)

    return movie_titles



def get_top_watched_movies(user_id):
    top_watched_movies = df[df.userId == user_id].sort_values(by='rating',ascending=False).movieId.head(5)
    return movies_df[movies_df.movieId.isin(top_watched_movies.values)]



