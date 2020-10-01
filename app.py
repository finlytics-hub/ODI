# import all the required libraries
from flask import Flask, request, url_for, redirect, render_template, jsonify
from sklearn.linear_model import LogisticRegression
import pandas as pd
import pickle
import numpy as np

# instantiate a Flask app
app = Flask(__name__)
# load our saved model
model = pickle.load(open('Final_LR_Model.pkl', 'rb'))
# list of input columns, i.e. the features that would be input by the user
cols = ['Country', 'Opposition', 'Home/Away', 'Country_Bowling Rank_1', 'Country_Bowling Rank_2', 'Country_Bowling Rank_3', 'Country_Bowling Rank_4', 'Country_Bowling Rank_5', 'Country_Bowling Rank_6', 'Country_Bowling Rank_7', 'Country_Bowling Rank_8', 'Country_Bowling Rank_9', 'Country_Bowling Rank_10', 'Country_Bowling Rank_11', 'Opposition_Bowling Rank_1', 'Opposition_Bowling Rank_2', 'Opposition_Bowling Rank_3', 'Opposition_Bowling Rank_4', 'Opposition_Bowling Rank_5', 'Opposition_Bowling Rank_6', 'Opposition_Bowling Rank_7', 'Opposition_Bowling Rank_8', 'Opposition_Bowling Rank_9', 'Opposition_Bowling Rank_10', 'Opposition_Bowling Rank_11', 'Country_Batting Rank_1', 'Country_Batting Rank_2', 'Country_Batting Rank_3', 'Country_Batting Rank_4', 'Country_Batting Rank_5', 'Country_Batting Rank_6', 'Country_Batting Rank_7', 'Country_Batting Rank_8', 'Country_Batting Rank_9', 'Country_Batting Rank_10', 'Country_Batting Rank_11', 'Opposition_Batting Rank_1', 'Opposition_Batting Rank_2', 'Opposition_Batting Rank_3', 'Opposition_Batting Rank_4', 'Opposition_Batting Rank_5', 'Opposition_Batting Rank_6', 'Opposition_Batting Rank_7', 'Opposition_Batting Rank_8', 'Opposition_Batting Rank_9', 'Opposition_Batting Rank_10', 'Opposition_Batting Rank_11', 'Win/Loss Ratio', 'batting average', 'bowling average', 'batting RPO', 'bowling RPO']
# list of all the dummy features as per our saved model
columns_list = ['Country_Bowling Rank_1', 'Country_Bowling Rank_2', 'Country_Bowling Rank_3', 'Country_Bowling Rank_4', 'Country_Bowling Rank_5', 'Country_Bowling Rank_6', 'Country_Bowling Rank_7', 'Country_Bowling Rank_8', 'Country_Bowling Rank_9', 'Country_Bowling Rank_10', 'Country_Bowling Rank_11', 'Opposition_Bowling Rank_1', 'Opposition_Bowling Rank_2', 'Opposition_Bowling Rank_3', 'Opposition_Bowling Rank_4', 'Opposition_Bowling Rank_5', 'Opposition_Bowling Rank_6', 'Opposition_Bowling Rank_7', 'Opposition_Bowling Rank_8', 'Opposition_Bowling Rank_9', 'Opposition_Bowling Rank_10', 'Opposition_Bowling Rank_11', 'Country_Batting Rank_1', 'Country_Batting Rank_2', 'Country_Batting Rank_3', 'Country_Batting Rank_4', 'Country_Batting Rank_5', 'Country_Batting Rank_6', 'Country_Batting Rank_7', 'Country_Batting Rank_8', 'Country_Batting Rank_9', 'Country_Batting Rank_10', 'Country_Batting Rank_11', 'Opposition_Batting Rank_1', 'Opposition_Batting Rank_2', 'Opposition_Batting Rank_3', 'Opposition_Batting Rank_4', 'Opposition_Batting Rank_5', 'Opposition_Batting Rank_6', 'Opposition_Batting Rank_7', 'Opposition_Batting Rank_8', 'Opposition_Batting Rank_9', 'Opposition_Batting Rank_10', 'Opposition_Batting Rank_11', 'Win/Loss Ratio', 'batting average', 'bowling average', 'batting RPO', 'bowling RPO', 'Country_Australia', 'Country_Bangladesh', 'Country_Bermuda', 'Country_England', 'Country_India', 'Country_Ireland', 'Country_Kenya', 'Country_Namibia', 'Country_Netherlands', 'Country_New Zealand', 'Country_Pakistan', 'Country_Scotland', 'Country_South Africa', 'Country_Sri Lanka', 'Country_West Indies', 'Country_Zimbabwe', 'Opposition_Australia', 'Opposition_Bangladesh', 'Opposition_Bermuda', 'Opposition_England', 'Opposition_India', 'Opposition_Ireland', 'Opposition_Kenya', 'Opposition_Namibia', 'Opposition_Netherlands', 'Opposition_New Zealand', 'Opposition_Pakistan', 'Opposition_Scotland', 'Opposition_South Africa', 'Opposition_Sri Lanka', 'Opposition_West Indies', 'Opposition_Zimbabwe', 'Home/Away_Home']

@app.route('/')
#load the main app page saved as 'home.html' in the 'templates' folder
def home():
    return render_template("home.html")

# method to make predictions
@app.route('/predict',methods=['POST'])
def predict():
    # obtain user input data as a list
    int_features = [x for x in request.form.values()]
    # save the playing country for which the user wants to predict the match outcome
    country = request.form['element_2']
    # convert the input list to a numpy array
    final = np.array(int_features)
    # convert the array to a DF
    unseen_data = pd.DataFrame([final], columns = cols)
    # create dummy variables
    unseen_data = pd.get_dummies(unseen_data, columns = ['Country', 'Opposition', 'Home/Away'], drop_first = True)
    # reindex the dummy encoded data to to ensure that all the same columns present therein as those in our logistic regression model
    unseen_data = unseen_data.reindex(labels = columns_list, axis=1, fill_value=0)
    # make predictions and save the probability of win
    result_prob = model.predict_proba(unseen_data)[: , 1]
    # convert the win probability to a float percentage
    result_prob_float = float(result_prob)*100
    # decision-boundary based on the best threshold as per our model
    if result_prob > 0.4827:
        result = 'Win'
    else:
        result = 'Lose'
    # display the 'result.html' page with our model-based prediction
    return render_template('result.html', pred=f'{country} is expected to {result} with a win probability of {result_prob_float:.2f}%')
  
# run the app
if __name__ == '__main__':
    app.run(debug=True)
