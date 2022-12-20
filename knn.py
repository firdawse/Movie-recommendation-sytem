import pandas as pd
import numpy as np

# Movies
movies = pd.read_csv('movie.csv')

# Ratings
rating = pd.read_csv('rating.csv')

# Réunir les deux bases de données
movies_merged = movies.merge(rating, on='movieId')

# Classer les données selon les titres les mieux notés
movies_average_rating = movies_merged.groupby('title')['rating'].mean().sort_values(ascending=False).reset_index().rename(columns={'rating':'Average Rating'})

# Classer les données selon le nombre de votes et le rating (Ascendant)
movies_rating_count = movies_merged.groupby('title')['rating'].count().sort_values(ascending=False).reset_index().rename(columns={'rating':'Rating Count'}) #ascending=False
movies_rating_count_avg = movies_rating_count.merge(movies_average_rating, on='title')

# Classer les données selon le nombre de votes et le rating (Descendant)
movies_rating_count2 = movies_merged.groupby('title')['rating'].count().sort_values(ascending=True).reset_index().rename(columns={'rating':'Rating Count'}) #ascending=False
movies_rating_count_avg2 = movies_rating_count2.merge(movies_average_rating, on='title')

"""Observations

* De nombreux films ont une note moyenne parfaite (5 étoiles) sur un ensemble de données de près de 100.000 notes d'utilisateurs. Cela suggère l'existence de valeurs aberrantes que nous devons confirmer davantage avec la visualisation.
* La présence de notes uniques pour plusieurs films suggère qu'on a défini une valeur seuil de notes pour produire de recommandations beaucoup plus précises

## 3) Visionnage des données

On se concentrerai sur la création de graphiques pour prouver/réfuter les observations et analyser les tendances que prendront les données (le cas échéant). On s'en tiendra à seaborn et matplotlib pour ce projet.
"""

# Sélection des meilleures notes
rating_with_RatingCount = movies_merged.merge(movies_rating_count, left_on = 'title', right_on = 'title', how = 'left')

# Description de la nouvelle base de données
pd.set_option('display.float_format', lambda x: '%.3f' % x)
print(rating_with_RatingCount['Rating Count'].describe())

# Extraction des films les plus populaires
popularity_threshold = 50
popular_movies= rating_with_RatingCount[rating_with_RatingCount['Rating Count']>=popularity_threshold]

"""
### Pivoter les Titres en tant qu'indices et userId en tant que colonnes (pour avoir une base de données prête à utiliser avec KNN)"""

import os
movie_features = popular_movies.pivot_table(index='title',columns='userId',values='rating').fillna(0)
# movie_features.to_excel('output.xlsx')

"""Ainsi, nous sauront quel utilisateur a noté quel film et aussi les films les plus notés

## 4) Implémentation du modèle KNN
"""

# Importations
from scipy.sparse import csr_matrix
movie_features_matrix = csr_matrix(movie_features.values)

# Entrainement du modèle
from sklearn.neighbors import NearestNeighbors
model_knn = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
model_knn.fit(movie_features_matrix)

 
def knn_predict(movie_id):
    # query_index = np.random.choice(movie_features.shape[0])
    query_index = np.uint64(movie_id)
    print("Recomendation for movie id:",query_index)
    distances, indices = model_knn.kneighbors(movie_features.iloc[query_index,:].values.reshape(1, -1), n_neighbors = 6)
    print("distances:",distances)
    print("indices",indices)

    out = []
    for i in range(0, len(distances.flatten())):
        if i == 0:
            out.append(movie_features.index[query_index]) 
        else:
            out.append(movie_features.index[indices.flatten()[i]])
    print(f"Out {out[0]}")
    print(f"Out {out[2]}")
    return out
