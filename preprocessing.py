import pandas as pd
import matplotlib.pyplot as plt
import chart_studio.plotly as py
import plotly.graph_objects as go
import seaborn as sns
import numpy as np
import random
import plotly
import plotly.express as px
import json
#datetime
from datetime import datetime


def getdataset():
    df50=pd.read_csv("NIFTY 50.csv")
    df50.head()
    df50.fillna(method="ffill", inplace=True)
    df50['year'] = pd.DatetimeIndex(df50['Date']).year
    df50['Date']=df50['Date'].apply(lambda x:datetime.strptime(x,'%Y-%m-%d'))
    df50['TrendValue'] = 0
    for i in range(1, len(df50['High'])):
        df50['TrendValue'][i] = df50['High'][i]-df50['High'][i-1]
        df50['Trend'] = 0

    i=0
    for val in df50['TrendValue']:
        if val>0:
            df50['Trend'][i] = 'Uptrend'
        elif val==0:
            df50['Trend'][i] = 'No change'
        else:
            df50['Trend'][i] = 'Downtrend'
    i = i+1
    return df50
    

df50 = getdataset()






