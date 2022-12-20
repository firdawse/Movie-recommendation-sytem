import pandas as pd 

movie = pd.read_csv("movie.csv")
movie = movie.loc[:,["movieId","title"]]
rating = pd.read_csv("rating.csv")
rating = rating.loc[:,["userId","movieId","rating"]]
# then merge movie and rating data
data = pd.merge(movie,rating)
data = data.iloc[:1000000,:]
# lets make a pivot table in order to make rows are users and columns are movies. And values are rating
pivot_table = data.pivot_table(index = ["userId"],columns = ["title"],values = "rating")

def movie(input):
    movie_watched = pivot_table[input]
    similarity_with_other_movies = pivot_table.corrwith(movie_watched)# find correlation between "Bad Boys (1995)" and other movies
    # find correlation between "Bad Boys (1995)" and other movies
    return similarity_with_other_movies.head().index.tolist()
