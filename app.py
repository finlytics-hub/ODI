from flask import Flask, request, url_for, redirect, render_template, jsonify
from sklearn.linear_model import LogisticRegression
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open('Final_LR_Model.pkl', 'rb'))
cols = ['Country', 'Opposition', 'Home/Away', 'Toss Won?', 'Ground', 'Match Month', 'Win/Loss Ratio', 'Country_Bowling Rank_1', 'Country_Bowling Rank_2', 'Country_Bowling Rank_3', 'Country_Bowling Rank_4', 'Country_Bowling Rank_5', 'Country_Bowling Rank_6', 'Country_Bowling Rank_7', 'Country_Bowling Rank_8', 'Country_Bowling Rank_9', 'Country_Bowling Rank_10', 'Country_Bowling Rank_11', 'Opposition_Bowling Rank_1', 'Opposition_Bowling Rank_2', 'Opposition_Bowling Rank_3', 'Opposition_Bowling Rank_4', 'Opposition_Bowling Rank_5', 'Opposition_Bowling Rank_6', 'Opposition_Bowling Rank_7', 'Opposition_Bowling Rank_8', 'Opposition_Bowling Rank_9', 'Opposition_Bowling Rank_10', 'Opposition_Bowling Rank_11', 'Country_Batting Rank_1', 'Country_Batting Rank_2', 'Country_Batting Rank_3', 'Country_Batting Rank_4', 'Country_Batting Rank_5', 'Country_Batting Rank_6', 'Country_Batting Rank_7', 'Country_Batting Rank_8', 'Country_Batting Rank_9', 'Country_Batting Rank_10', 'Country_Batting Rank_11', 'Opposition_Batting Rank_1', 'Opposition_Batting Rank_2', 'Opposition_Batting Rank_3', 'Opposition_Batting Rank_4', 'Opposition_Batting Rank_5', 'Opposition_Batting Rank_6', 'Opposition_Batting Rank_7', 'Opposition_Batting Rank_8', 'Opposition_Batting Rank_9', 'Opposition_Batting Rank_10', 'Opposition_Batting Rank_11']
columns_list = ['Match Month', 'Win/Loss Ratio', 'Country_Bowling Rank_1', 'Country_Bowling Rank_2', 'Country_Bowling Rank_3', 'Country_Bowling Rank_4', 'Country_Bowling Rank_5', 'Country_Bowling Rank_6', 'Country_Bowling Rank_7', 'Country_Bowling Rank_8', 'Country_Bowling Rank_9', 'Country_Bowling Rank_10', 'Country_Bowling Rank_11', 'Opposition_Bowling Rank_1', 'Opposition_Bowling Rank_2', 'Opposition_Bowling Rank_3', 'Opposition_Bowling Rank_4', 'Opposition_Bowling Rank_5', 'Opposition_Bowling Rank_6', 'Opposition_Bowling Rank_7', 'Opposition_Bowling Rank_8', 'Opposition_Bowling Rank_9', 'Opposition_Bowling Rank_10', 'Opposition_Bowling Rank_11', 'Country_Batting Rank_1', 'Country_Batting Rank_2', 'Country_Batting Rank_3', 'Country_Batting Rank_4', 'Country_Batting Rank_5', 'Country_Batting Rank_6', 'Country_Batting Rank_7', 'Country_Batting Rank_8', 'Country_Batting Rank_9', 'Country_Batting Rank_10', 'Country_Batting Rank_11', 'Opposition_Batting Rank_1', 'Opposition_Batting Rank_2', 'Opposition_Batting Rank_3', 'Opposition_Batting Rank_4', 'Opposition_Batting Rank_5', 'Opposition_Batting Rank_6', 'Opposition_Batting Rank_7', 'Opposition_Batting Rank_8', 'Opposition_Batting Rank_9', 'Opposition_Batting Rank_10', 'Opposition_Batting Rank_11', 'Country_Australia', 'Country_Bangladesh', 'Country_Bermuda', 'Country_England', 'Country_India', 'Country_Ireland', 'Country_Kenya', 'Country_Namibia', 'Country_Netherlands', 'Country_New Zealand', 'Country_Pakistan', 'Country_Scotland', 'Country_South Africa', 'Country_Sri Lanka', 'Country_West Indies', 'Country_Zimbabwe', 'Opposition_Australia', 'Opposition_Bangladesh', 'Opposition_Bermuda', 'Opposition_England', 'Opposition_India', 'Opposition_Ireland', 'Opposition_Kenya', 'Opposition_Namibia', 'Opposition_Netherlands', 'Opposition_New Zealand', 'Opposition_Pakistan', 'Opposition_Scotland', 'Opposition_South Africa', 'Opposition_Sri Lanka', 'Opposition_West Indies', 'Opposition_Zimbabwe', 'Home/Away_Home', 'Toss Won?_Yes', 'Ground_Abu Dhabi', 'Ground_Adelaide', 'Ground_Ahmedabad', 'Ground_Albion', 'Ground_Amstelveen', 'Ground_Auckland', 'Ground_Basseterre', 'Ground_Belfast', 'Ground_Bengaluru', 'Ground_Benoni', 'Ground_Birmingham', 'Ground_Bloemfontein', 'Ground_Bogra', 'Ground_Bridgetown', 'Ground_Brisbane', 'Ground_Bristol', 'Ground_Bulawayo', 'Ground_Canberra', 'Ground_Cape Town', 'Ground_Cardiff', 'Ground_Centurion', 'Ground_Chandigarh', 'Ground_Chattogram', 'Ground_Chennai', 'Ground_Chester-le-Street', 'Ground_Christchurch', 'Ground_Colombo (PSS)', 'Ground_Colombo (RPS)', 'Ground_Colombo (SSC)', 'Ground_Cuttack', 'Ground_Dambulla', 'Ground_Delhi', 'Ground_Dhaka', 'Ground_Dubai (DSC)', 'Ground_Dublin', 'Ground_Dublin (Malahide)', 'Ground_Dunedin', 'Ground_Durban', 'Ground_East London', 'Ground_Edinburgh', 'Ground_Faisalabad', 'Ground_Faridabad', 'Ground_Fatullah', 'Ground_Galle', 'Ground_Georgetown', 'Ground_Greater Noida', 'Ground_Gros Islet', 'Ground_Gujranwala', 'Ground_Guwahati', 'Ground_Gwalior', 'Ground_Hambantota', 'Ground_Hamilton', 'Ground_Harare', 'Ground_Hobart', 'Ground_Hyderabad (Deccan)', 'Ground_Hyderabad (Sind)', 'Ground_Indore', 'Ground_Jaipur', 'Ground_Jamshedpur', 'Ground_Johannesburg', 'Ground_Kandy', 'Ground_Kanpur', 'Ground_Karachi', 'Ground_Kimberley', 'Ground_Kingston', 'Ground_Kingstown', 'Ground_Kochi', 'Ground_Kolkata', 'Ground_Kuala Lumpur', 'Ground_Lahore', 'Ground_Leeds', 'Ground_Lord\'s', 'Ground_Manchester', 'Ground_Margao', 'Ground_Melbourne', 'Ground_Melbourne (Docklands)', 'Ground_Mohali', 'Ground_Mombasa', 'Ground_Moratuwa', 'Ground_Mount Maunganui', 'Ground_Multan', 'Ground_Mumbai', 'Ground_Mumbai (BS)', 'Ground_Nagpur', 'Ground_Nairobi (Gym)', 'Ground_Napier', 'Ground_Nelson', 'Ground_North Sound', 'Ground_Nottingham', 'Ground_Others', 'Ground_Paarl', 'Ground_Pallekele', 'Ground_Perth', 'Ground_Peshawar', 'Ground_Port Elizabeth', 'Ground_Port of Spain', 'Ground_Potchefstroom', 'Ground_Providence', 'Ground_Pune', 'Ground_Queenstown', 'Ground_Rajkot', 'Ground_Rawalpindi', 'Ground_Rotterdam', 'Ground_Sharjah', 'Ground_Sialkot', 'Ground_Singapore', 'Ground_Southampton', 'Ground_St George\'s', 'Ground_St John\'s', 'Ground_Sydney', 'Ground_Tangier', 'Ground_Taunton', 'Ground_The Hague', 'Ground_The Oval', 'Ground_Toronto', 'Ground_Vadodara', 'Ground_Visakhapatnam', 'Ground_Wellington']

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/predict',methods=['POST'])
def predict():
    int_features = [x for x in request.form.values()]
    final = np.array(int_features)
    unseen_data = pd.DataFrame([final], columns = cols)
    unseen_data = pd.get_dummies(unseen_data, columns = ['Country', 'Opposition', 'Home/Away', 'Toss Won?', 'Ground'], drop_first = True)
    unseen_data = unseen_data.reindex(labels = columns_list, axis=1, fill_value=0)
    result_prob = model.predict_proba(unseen_data)[: , 1]
    if result_prob > 0.5:
        result = 'Win'
    else:
        result = 'Lose'
    return render_template('home.html', pred='Country is expected to {}'.format(result))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    unseen_data = pd.DataFrame([data])
    unseen_data = pd.get_dummies(unseen_data, columns = ['Country', 'Opposition', 'Home/Away', 'Toss Won?', 'Ground'], drop_first = True)
    unseen_data = unseen_data.reindex(labels = columns_list, axis=1, fill_value=0)
    result_prob = model.predict_proba(unseen_data)[: , 1]
    if result_prob > 0.5:
        result = 'Win'
    else:
        result = 'Lose'
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
