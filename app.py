from flask import Flask, render_template, request, flash
import model as m
import json, requests
from mlp import predict,get_top_watched_movies
from knn import knn_predict
from kmeans import kmeans_predict

app = Flask(__name__)

app.secret_key = "manbearpig_MUDMAN888"

@app.route("/hello")
def index():
	return render_template("welcome.html")

@app.route("/result1", methods=['POST', 'GET'])
def result1():
	# flash(m.movie(request.form['name_input']))
	predicted_movies = m.movie(request.form['user_id'])

	if predicted_movies == 1:
		flash("User not found", "error")
		return render_template("index.html", movies = [], poster_urls=[])

	poster_urls = []
	for movie in predicted_movies:
		response = requests.get(f"https://api.themoviedb.org/3/search/movie?api_key=15d2ea6d0dc1d476efbca3eba2b9bbfb&query='{movie[:-6]}'")
		if response.json()['total_results'] == 0:
			poster_urls.append('')
		else:
			response = response.json()["results"][0]['poster_path']
			print(response)
			poster_urls.append(f"http://image.tmdb.org/t/p/w500/{response}")
	
	return render_template("index.html", movies = predicted_movies, poster_urls = poster_urls)

@app.route('/UserBased')
def UserBased():
	return render_template('index.html', movies = [], poster_urls = [])

@app.route('/kmeans')
def kmeans():
	return render_template('kmeans.html', movies = [], poster_urls = [])

@app.route("/result2", methods=['POST', 'GET'])
def result2():
	predicted_movies = kmeans_predict(request.form['user_id'])

	if predicted_movies == 1:
		flash("User not found", "error")
		return render_template("kmeans.html", movies = [])

	poster_urls = []
	for movie in predicted_movies:
		response = requests.get(f"https://api.themoviedb.org/3/search/movie?api_key=15d2ea6d0dc1d476efbca3eba2b9bbfb&query='{movie[:-6]}'")
		if response.json()['total_results'] == 0:
			poster_urls.append('')
		else:
			response = response.json()["results"][0]['poster_path']
			print(response)
			poster_urls.append(f"http://image.tmdb.org/t/p/w500/{response}")
	
	return render_template("kmeans.html", movies = predicted_movies, poster_urls = poster_urls)


@app.route('/knn')
def knn():
	return render_template('knn.html', movies = [], poster_urls = [])

@app.route("/result3", methods=['POST', 'GET'])
def result3():
	predicted_movies = knn_predict(request.form['user_id'])

	if predicted_movies == 1:
		flash("User not found", "error")
		return render_template("knn.html", movies = [])

	poster_urls = []
	for movie in predicted_movies:
		response = requests.get(f"https://api.themoviedb.org/3/search/movie?api_key=15d2ea6d0dc1d476efbca3eba2b9bbfb&query='{movie[:-6]}'")
		if response.json()['total_results'] == 0:
			poster_urls.append('')
		else:
			response = response.json()["results"][0]['poster_path']
			print(response)
			poster_urls.append(f"http://image.tmdb.org/t/p/w500/{response}")
	
	return render_template("knn.html", movies = predicted_movies, poster_urls = poster_urls)

@app.route('/mlp')
def mlp():
	return render_template('mlp.html',movies = [], watched_movies = [], poster_urls = [], watched_poster_urls=[])

@app.route("/result4", methods=['POST', 'GET'])
def result4():
	predicted_movies = predict(request.form['user_id'])
	watched_movies = get_top_watched_movies(request.form['user_id'])

	if predicted_movies == 1:
		flash("User not found", "error")
		return render_template("mlp.html", movies = [], watched_movies = [], watched_poster_urls=[])

	poster_urls = []
	for movie in predicted_movies:
		response = requests.get(f"https://api.themoviedb.org/3/search/movie?api_key=15d2ea6d0dc1d476efbca3eba2b9bbfb&query='{movie[:-6]}'")
		if response.json()['total_results'] == 0:
			poster_urls.append('')
		else:
			response = response.json()["results"][0]['poster_path']
			print(response)
			poster_urls.append(f"http://image.tmdb.org/t/p/w500/{response}")

	watched_poster_urls = []
	for movie in watched_movies:
		response = requests.get(f"https://api.themoviedb.org/3/search/movie?api_key=15d2ea6d0dc1d476efbca3eba2b9bbfb&query='{movie[:-6]}'")
		if response.json()['total_results'] == 0:
			watched_poster_urls.append('')
		else:
			response = response.json()["results"][0]['poster_path']
			print(response)
			watched_poster_urls.append(f"http://image.tmdb.org/t/p/w500/{response}")			
	return render_template("mlp.html", movies = predicted_movies, watched_movies = watched_movies, poster_urls = poster_urls, watched_poster_urls=watched_poster_urls)


if __name__ == '__main__':
    app.run(port=4000,debug=True)


