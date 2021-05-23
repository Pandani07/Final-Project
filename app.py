from flask import Flask, redirect, url_for, request, render_template
import pickle
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import chart_studio.plotly as py
import plotly.graph_objects as go
from IPython.display import HTML
import seaborn as sns
import numpy as np
import random
import plotly
import plotly.express as px
import json
from preprocessing import getdataset


df50 = getdataset()

with open('pickle_model.pkl', 'rb') as file:
    pickle_model = pickle.load(file)
    
app = Flask(__name__)


posts = [
    {
        'author':'Anirudh Dutt',
        'title':'Stock Prediction',
        'content': 'First post',
        'date_posted':'April 2021'
    },
    {
        'author':'Anirudh Dutt',
        'title':'Stock Prediction',
        'content': 'second post',
        'date_posted':'may 2021'
    }
]


@app.route('/')
def index():
    return render_template("index.html")

@app.route('/trend', methods=['GET'])
def trend():
    return render_template("trend.html")


@app.route('/trend', methods=['POST'])
def detect_trend():
    rsi = request.form['rsi']
    crc = request.form['crc']
    obv = request.form['obv']
    vwap = request.form['vwap']
    res = pickle_model.predict([[rsi, crc, obv, vwap]])
    print(res)
    return render_template('detect_trend.html', data=res)

# @app.route('/send', methods=['GET', 'POST'])
# def predict():
#     if request.method== 'POST':
#         prediction = predict_price()
    
#     if prediction!=0:
#         return render_template("index.html", prediction = prediction)

@app.route('/visualize', methods=['GET', 'POST'])
def visualize():
    fig=px.line(df50,x='Date',y='Open', title='Nifty 50 Closing Price vs Date')
    image = HTML(fig.to_html())
    return render_template("visualize.html",image=image)

@app.route('/projectlink', methods=['GET', 'POST'])
def projectlink():
    return render_template("projectlink.html", posts=posts)


if __name__ == "__main__":
    app.run(debug=True)