from flask import Flask, redirect, url_for, request, render_template
import pickle
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import chart_studio.plotly as py
import plotly.graph_objects as go
from IPython.display import HTML
import plotly.express as px
from preprocessing import getdataset
# from getrsi import getrsi
import pickle
#from indicators  import buildclassifier

  


df50 = getdataset()

with open('pickle_model.pkl', 'rb') as file:
    pickle_model = pickle.load(file)
    
app = Flask(__name__)


@app.route('/')
def index():
    return render_template("index.html")

@app.route('/trend', methods=['GET'])
def trend():
    return render_template("trend.html")


@app.route('/trend', methods=['POST'])
def detect_trend():
    
    company = request.form['company']

    print()

    # rsi = getrsi()
    # res = pickle_model.predict([[rsi]])
    # trend_value=str(res).strip('[]')
    # print(trend_value)
    # if trend_value=='1':
    #     trend ='Uptrend'
    # else:
    #     trend = 'Downtrend'
    # return render_template('trend.html', rsi=rsi, trend=trend)

    return render_template('trend.html')


@app.route('/visualize', methods=['GET', 'POST'])
def visualize():
    fig=px.line(df50, x='Date',y='Open', title='Nifty 50 Closing Price vs Date')
    image = HTML(fig.to_html())
    return render_template("visualize.html",image=image)

@app.route('/projectlink', methods=['GET', 'POST'])
def projectlink():
    project_link = 'https://github.com/Pandani07/Final-Project'
    return render_template("projectlink.html", project_link = project_link)



if __name__ == "__main__":
    app.run(debug=True)