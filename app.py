from flask import Flask, render_template, request, flash
import model as m
from mlp import predict

app = Flask(__name__)

app.secret_key = "manbearpig_MUDMAN888"

@app.route("/hello")
def index():
	return render_template("welcome.html")

@app.route("/result1", methods=['POST', 'GET'])
def result1():
	flash(m.movie(request.form['name_input']))
	return render_template("index.html")

@app.route('/UserBased')
def UserBased():
	return render_template('index.html')

@app.route('/kmeans')
def kmeans():
	return render_template('kmeans.html')

@app.route("/result2", methods=['POST', 'GET'])
def result2():
	flash(m.movie(request.form['name_input']))
	return render_template("kmeans.html")

@app.route('/knn')
def knn():
	return render_template('knn.html')

@app.route("/result3", methods=['POST', 'GET'])
def result3():
	flash(m.movie(request.form['name_input']))
	return render_template("knn.html")

@app.route('/mlp')
def mlp():
	return render_template('mlp.html')

@app.route("/result4", methods=['POST', 'GET'])
def result4():
	movies = predict(request.form['name_input'])
	return render_template("mlp.html",movies = movies)


if __name__ == '__main__':
    app.run(port=4000,debug=True)


