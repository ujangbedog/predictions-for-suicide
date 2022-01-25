import numpy as np
import sklearn
import requests
from flask import Flask, request, jsonify, render_template
import joblib


app = Flask(__name__)
model = joblib.load(open('model.pkl', 'rb'))


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/suicide')
def suicide():
    return render_template('suicide.html')


@app.route("/predict", methods=['POST'])
def predict():

    if request.method == 'POST':

        gender = int(request.form['gender'])
        sexuallity = int(request.form['sexuallity'])
        friends = int(request.form['friends'])
        age = int(request.form['age'])
        income = int(request.form['income'])
        bodyweight = int(request.form['bodyweight'])
        virgin = int(request.form['virgin'])
        social_fear = int(request.form['social_fear'])
        depressed = int(request.form['depressed'])
        employment = int(request.form['employment'])

        
        prediction=model.predict([[gender, sexuallity, friends, age, income, bodyweight, virgin, social_fear, employment, depressed]])
        percent=model.predict_proba([[gender, sexuallity, friends, age, income, bodyweight, virgin, social_fear, employment, depressed]])
        output=round(prediction[0],2)
        if output ==1:
            return render_template('result-yes-suicide.html', percent=percent[0][1]*100)
        else:
            return render_template('result-not-suicide.html', percent=percent[0][1]*100)
    

if __name__=="__main__":
    #app.config['DEBUG'] = true
    app.run()