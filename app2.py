from flask import Flask, request, url_for, redirect, render_template, jsonify
from pycaret.classification import *
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)

model = load_model('PyCaret_LGBM_Model')
cols = ['Country', 'Opposition', 'Home/Away', 'Toss Won?', 'Ground', 'Win/Loss Ratio', 'Match Month', 'Bowling Rank_1', 'Bowling Rank_2', 'Bowling Rank_3', 'Bowling Rank_4', 'Bowling Rank_5', 'Bowling Rank_6', 'Bowling Rank_7', 'Bowling Rank_8', 'Bowling Rank_9', 'Bowling Rank_10', 'Bowling Rank_11', 'Batting Rank_1', 'Batting Rank_2', 'Batting Rank_3', 'Batting Rank_4', 'Batting Rank_5', 'Batting Rank_6', 'Batting Rank_7', 'Batting Rank_8', 'Batting Rank_9', 'Batting Rank_10', 'Batting Rank_11']

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/predict',methods=['POST'])
def predict():
    int_features = [x for x in request.form.values()]
    final = np.array(int_features)
    data_unseen = pd.DataFrame([final], columns = cols)
    prediction = predict_model(model, data=data_unseen)
    prediction = prediction.Label[0]
    return render_template('home.html', pred='Expected Winner will be {}'.format(prediction))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    data_unseen = pd.DataFrame([data])
    prediction = predict_model(model, data=data_unseen)
    output = prediction.Label[0]
    return jsonify(output)

if __name__ == '__main__':
    app.run(debug=True)