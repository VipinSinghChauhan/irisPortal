from flask import Flask, render_template, request, redirect, url_for
from joblib import load
import numpy as np
from sklearn.datasets import load_iris
import os

iris = load_iris()

model = load('iris.joblib')

def predictPlant(data):
    return model.predict(data)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html',output=url_for('static', filename='../static/flower.jpg'),plant="iris-specie",error="")

@app.route('/', methods=['POST', 'GET'])
def get_data():
    if request.method == 'POST':
        try:
            sepal_length = float(request.form['sepal_length'])
            sepal_width = float(request.form['sepal_width'])
            petal_length = float(request.form['petal_length'])
            petal_width = float(request.form['petal_width'])
            data = np.array([[sepal_length,sepal_width,petal_length,petal_width]])
            plant = iris.target_names[model.predict(data)[0]]
        except:
            return render_template('index.html',output=url_for('static', filename='../static/sad_flower.jpg'),plant="Wrong info",error="Text values not allowed :(")
        # print(predictPlant(data))

        return render_template('index.html',output=url_for('static', filename='../static/'+str(plant)+'.jpg'),plant=plant,error="")

if __name__ == '__main__' :
    app.run(debug=True)