#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
import pickle
from sys import exc_info
import ast


# In[ ]:


ratings = pd.read_csv('rating.csv')


# In[ ]:


ratings = ratings[ratings['rating'] >= 4.0]


# In[ ]:


ratings = ratings[:100000]


# In[ ]:


def moviesListForUsers(users, users_data):
    # users = a list of users IDs
    # users_data = a df of users favourite movies or users watched movies
    
    users_movies_list = []
    for user in users :
        
        # 1. get the the movieId of the corresponding user
        user_movie = users_data[users_data['userId'] == user]['movieId']
        
        # 2. Convert it to a list
        user_movie = list(user_movie)
        
        # 3. convert list elements to a string
        user_movie = str(user_movie)
        
        # 4. Split the list by '[' and ']'
        user_movie = user_movie.split('[')[1].split(']')[0]
        
        # 5. Append them to the returned list
        users_movies_list.append(user_movie)
        
    return users_movies_list


# In[ ]:


users = np.unique(users_fav_movies['userId'])


# In[ ]:


users_movies_list = moviesListForUsers(users, users_fav_movies)


# In[ ]:


def prepSparseMatrix(list_of_str):
    cv = CountVectorizer(token_pattern = r'[^\,\ ]+', lowercase = False)
    sparseMatrix = cv.fit_transform(list_of_str)
    return sparseMatrix.toarray(), cv.get_feature_names()


# In[ ]:


sparseMatrix, feature_names = prepSparseMatrix(users_movies_list)


# In[ ]:


df_sparseMatrix = pd.DataFrame(sparseMatrix, index = users, columns = feature_names)


# In[ ]:


kmeans = KMeans(n_clusters=2, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
clusters = kmeans.fit_predict(sparseMatrix)


# In[ ]:


users_cluster = pd.DataFrame(np.concatenate((users.reshape(-1,1), clusters.reshape(-1,1)), axis = 1), columns = ['userId', 'Cluster'])


# In[ ]:


# For this, first we’ll define a method for creating movies of clusters.

def clustersMovies(users_cluster, users_data):
    clusters = list(users_cluster['Cluster'])
    each_cluster_movies = list()
    for i in range(len(np.unique(clusters))):
        users_list = list(users_cluster[users_cluster['Cluster'] == i]['userId'])
        users_movies_list = list()
        for user in users_list:    
            users_movies_list.extend(list(users_data[users_data['userId'] == user]['movieId']))
        users_movies_counts = list()
        users_movies_counts.extend([[movie, users_movies_list.count(movie)] for movie in np.unique(users_movies_list)])
        each_cluster_movies.append(pd.DataFrame(users_movies_counts, columns=['movieId', 'Count']).sort_values(by = ['Count'], ascending = False).reset_index(drop=True))
    return each_cluster_movies


# In[ ]:


cluster_movies = clustersMovies(users_cluster, users_fav_movies)


# In[ ]:


class saveLoadFiles:
    def save(self, filename, data):
        try:
            file = open(filename + '.pkl', 'wb')
            pickle.dump(data, file)
        except:
            err = 'Error: {0}, {1}'.format(exc_info()[0], exc_info()[1])
            print(err)
            file.close()
            return [False, err]
        else:
            file.close()
            return [True]
    def load(self, filename):
        try:
            file = open(filename + '.pkl', 'rb')
        except:
            err = 'Error: {0}, {1}'.format(exc_info()[0], exc_info()[1])
            print(err)
            file.close()
            return [False, err]
        else:
            data = pickle.load(file)
            file.close()
            return data
    def loadClusterMoviesDataset(self):
        return self.load('clusters_movies_dataset')
    def saveClusterMoviesDataset(self, data):
        return self.save('clusters_movies_dataset', data)
    def loadUsersClusters(self):
        return self.load('users_clusters')
    def saveUsersClusters(self, data):
        return self.save('users_clusters', data)


# In[ ]:


saveLoadFile = saveLoadFiles()
print(saveLoadFile.saveClusterMoviesDataset(cluster_movies))
print(saveLoadFile.saveUsersClusters(users_cluster))


# In[ ]:


def getMoviesOfUser(user_id, users_data):
    return list(users_data[users_data['userId'] == user_id]['movieId'])


# In[ ]:


class userRequestedFor:
    def __init__(self, user_id, users_data):
        self.users_data = users_data.copy()
        self.user_id = user_id

        # Find User Cluster
        users_cluster = saveLoadFiles().loadUsersClusters()
        self.user_cluster = int(users_cluster[users_cluster['userId'] == self.user_id]['Cluster'])

        # Load User Cluster Movies Dataframe
        self.movies_list = saveLoadFiles().loadClusterMoviesDataset()
        self.cluster_movies = self.movies_list[self.user_cluster] # dataframe
        self.cluster_movies_list = list(self.cluster_movies['movieId']) # list
        
    def updatedFavouriteMoviesList(self, new_movie_Id):
        if new_movie_Id in self.cluster_movies_list:
            self.cluster_movies.loc[self.cluster_movies['movieId'] == new_movie_Id, 'Count'] += 1
        else:
            self.cluster_movies = self.cluster_movies.append([{'movieId':new_movie_Id, 'Count': 1}], ignore_index=True)
            
        self.cluster_movies.sort_values(by = ['Count'], ascending = False, inplace= True)
        self.movies_list[self.user_cluster] = self.cluster_movies
        saveLoadFiles().saveClusterMoviesDataset(self.movies_list)

    def recommendMostFavouriteMovies(self):
        try:
            user_movies = getMoviesOfUser(self.user_id, self.users_data)
            cluster_movies_list = self.cluster_movies_list.copy()
            for user_movie in user_movies:
                if user_movie in cluster_movies_list:
                    cluster_movies_list.remove(user_movie)
            return [True, cluster_movies_list]
        except KeyError:
            err = "User history does not exist"
            print(err)
            return [False, err]
        except:
            err = 'Error: {0}, {1}'.format(exc_info()[0], exc_info()[1])
            print(err)
            return [False, err]


# In[ ]:


movies_metadata = pd.read_csv('movie.csv')


# In[ ]:


userRequestedFor = userRequestedFor(user_id, users_fav_movies)


# In[ ]:


_, cluster_movies_list = userRequestedFor.recommendMostFavouriteMovies()


# In[ ]:


userRecommendedMovies = list(movies_metadata[movies_metadata['movieId'].isin(cluster_movies_list)]['title'])

