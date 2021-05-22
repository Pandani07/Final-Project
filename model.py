import pandas as pd
import matplotlib.pyplot as plt
import chart_studio.plotly as py
import plotly.graph_objects as go
import seaborn as sns
import numpy as np
import random
#datetime
from datetime import datetime

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
from tensorflow import keras
from tensorflow.keras.models import load_model

from sklearn.model_selection import GridSearchCV, KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestRegressor
rf  = RandomForestRegressor(n_estimators=100)

print("Imported all libraries")

df50=pd.read_csv("NIFTY 50.csv")
df50.head()
df50.fillna(method="ffill", inplace=True)
df50['year'] = pd.DatetimeIndex(df50['Date']).year

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


x1 = df50.iloc[:, 4].values.reshape(-1, 1)
print(x1.shape)


x = df50.iloc[:, 4].values.reshape(-1, 1)
y = df50.iloc[:, 7].values.reshape(-1, 1)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=0.25)

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
y_train = scaler.fit_transform(y_train)
y_test = scaler.transform(y_test)

print("X_train shape: {}\n".format(x_train.shape))
print("X_test shape: {}\n".format(x_test.shape))
print("Y_train shape: {}\n".format(y_train.shape))
print("Y_test shape: {}\n".format(y_test.shape))

rf.fit(x_train, y_train.ravel())

y_pred = rf.predict(x_test)

print("Score: {}".format(str(rf.score(y_test, y_pred))))

forecast_days = 60

def getdata(df50):
    training_set = df50.iloc[0:4000, 1:2].values
    training_set_scaled = scaler.fit_transform(training_set)
    x_train = []
    y_train = []

    for i in range(forecast_days, len(training_set_scaled)):
        x_train.append(training_set_scaled[i-forecast_days:i, 0])
        y_train.append(training_set_scaled[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)

    return x_train, y_train

x_train, y_train = getdata(df50)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

model = load_model('model_100.h5')

print(model.summary())


def get_testdata(df50):

    dataset_test = df50[4000:df50['Close'].shape[0]]
    real_stock_price = dataset_test.iloc[0:len(dataset_test), 1:2].values
    dataset_total = df50['Open']
    inputs = dataset_total[len(dataset_total) - len(dataset_test) - forecast_days:].values
    inputs  = inputs.reshape(-1, 1)
    inputs = scaler.transform(inputs)

    return inputs, real_stock_price
    

inputs, real_stock_price = get_testdata(df50)


x_test = []
for i in range(forecast_days,  len(inputs)):
    x_test.append(inputs[i-forecast_days:i, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))


result = model.predict(x_test)
result_set = scaler.inverse_transform(result)


def print_shapes():
    
    print("X_train shape: {}\n".format(x_train.shape))
    print("X_test shape: {}\n".format(x_test.shape))
    print("Y_train shape: {}\n".format(y_train.shape))
    print("Y_test shape: {}\n".format(real_stock_price.shape))
    print("Result shape: {}\n".format(result_set.shape))

print_shapes()


def prediction_plot():
    plt.plot(real_stock_price, color="red", label = 'Real NIFTY 50 Stock Price')
    plt.plot(result_set, color="blue", label="Predicted Price")
    plt.xlabel("Time")
    plt.ylabel("Stock Price")
    plt.legend()
    plt.show()

prediction_plot()


from sklearn.metrics import accuracy_score, confusion_matrix, mean_absolute_error, r2_score

mae = mean_absolute_error(real_stock_price, result_set)
r2=r2_score(real_stock_price, result_set)

print("Mean Absolute Error: {}".format(mae))

print("R2 Score: {}".format(r2))



minimum = int(min(df50['Open']))
maximum = int(max(df50['Open']))


def get_errors():
    
    errors = []

    for i in range(0, len(real_stock_price)):
        errors.append(result_set[i]-real_stock_price[i])
    
    mse = np.square(errors).mean()
    rmse= np.sqrt(mse)

    print("Mean Squared Error: {}\n".format(mse))
    print("Root Mean Squared Error: {}\n".format(rmse))

get_errors()


def predict_price(period):
    real_data = [inputs[len(inputs)+1-period: len(inputs+1), 0]]
    real_data = np.array(real_data)
    real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))
    prediction = model.predict(real_data)
    prediction = scaler.inverse_transform(prediction)
    return prediction
    
#RSI
df50['TrendValue'][0] = np.nan
upday, downday = df50.copy(), df50.copy()
upday.loc['TrendValue'] = upday.loc[(upday['TrendValue']<0), 'TrendValue' ] = 0
downday.loc['TrendValue'] = downday.loc[(downday['TrendValue']>0), 'TrendValue' ] = 0
downday['TrendValue'] = downday['TrendValue'].abs()
ewmup = upday['TrendValue'].transform(lambda x: x.ewm(span = 20).mean())
ewmdown = downday['TrendValue'].transform(lambda x: x.ewm(span = 20).mean())
relative_strength = ewmup/ewmdown
RSI = (100.0 - (1.0/100+relative_strength))

df50['Up Days'] = upday['TrendValue']
df50['Down Days'] = downday['TrendValue']
df50['RSI'] = RSI
df50['Prediction'] = np.nan

#Price Rate of Change
df50['Close Rateofchange'] = df50['Close'].transform(lambda x: x.pct_change(periods = 20))

#On Balance Volume
obv= []
obv.append(0)
for i in range(1, len(df50['Close'])):
    if df50['Close'][i] > df50['Close'][i-1]:
        obv.append( obv[-1] + df50['Volume'][i])
    elif df50['Close'][i] < df50['Close'][i-1]:
        obv.append( obv[-1] - df50['Volume'][i])
    else:
        obv.append(obv[-1])
df50['OBV'] = obv

#Volume weighted average price
close_mean = df50['Close'].mean()
print("Mean closing price: {:.2f} Rs".format(close_mean))

period = 20
vwap = []
for i in range(0,len(df50)):
        if(i>=period):
            numerator = sum(df50["Close"][i-period:i]*df50["Volume"][i-period:i])
            denomenator = sum(df50["Volume"][i-period:i])
            vwap.append(numerator/denomenator)
        else:
            vwap.append(None)

df50['VWAP'] = vwap
df50['Prediction Value'] = 0

for i in range(len(df50)):
    if df50['Up Days'][i]>0:
        df50['Prediction Value'][i] = 1
    elif df50['Down Days'][i]>0:
        df50['Prediction Value'][i] = 0
        


from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=100, criterion='gini', oob_score=True)

df = df50.copy()
df.drop(columns=['Prediction'], inplace=True)
df.dropna(how='any', inplace=True)

x = df[['RSI', 'Close Rateofchange','OBV','VWAP']]
y = df[['Prediction Value']]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=0)
classifier.fit(x_train, y_train.values.ravel())
y_pred = classifier.predict(x_test)

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
print("Accuracy of the Classifier: {}\n".format(accuracy_score(y_test, y_pred)))
print("Confusion Matrix:\n")
cm = confusion_matrix(y_test, y_pred)
print(cm)
print("\n")
print("Classification Report:\n")
cr = classification_report(y_test, y_pred)
print(cr)

real_data1 = []
for i in range(0, forecast_days):
    ele = random.randint(minimum, maximum)
    real_data1.append(ele)

real_data = np.array(real_data1).reshape(-1,1)
real_data = scaler.fit_transform(real_data)
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

def printans(prediction):
    answer = str(prediction).strip("[[]]")
    print("The model predicts {} as the forecast for the next day ".format(answer))

prediction = model.predict(real_data)
prediction = scaler.inverse_transform(prediction)
printans(prediction)


#send last 2 months data