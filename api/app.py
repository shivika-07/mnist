import pickle
from flask import Flask
from flask import request
from joblib import dump, load
import numpy as np
import os


app = Flask(__name__)

bestmodel = "./model_svm.joblib"
decbestmodel = "./model_decisiontree.joblib"

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/predict", methods=['POST'])
def predict():
    clf = load("model_svm.joblib")
    inputjason = request.json
    image = inputjason['image']
    print(image)
    image = np.array(image).reshape(1,-1)
    predicted = clf.predict(image)
    return str(predicted[0])


@app.route("/decpredict", methods=['POST'])
def decpredict():
    clf = load("model_decisiontree.joblib")
    inputjason = request.json
    image = inputjason['image']
    # print(image)
    image = np.array(image).reshape(1,-1)
    predicted = clf.predict(image)
    return str(predicted[0])