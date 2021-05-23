import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
from tensorflow import keras
from tensorflow.keras.models import load_model

from sklearn.model_selection import GridSearchCV, KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestRegressor
rf  = RandomForestRegressor(n_estimators=100)

from preprocessing import getdataset
df50 = getdataset()

def randomforestregressorpart():

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
    
randomforestregressorpart()
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

def loadingnn():
    model = load_model('model_100.h5')
    return model

model = loadingnn()
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

def formattestingdata():
    x_test = []
    for i in range(forecast_days,  len(inputs)):
        x_test.append(inputs[i-forecast_days:i, 0])
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    return x_test

x_test = formattestingdata()
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

def printscores():
    mae = mean_absolute_error(real_stock_price, result_set)
    r2=r2_score(real_stock_price, result_set)
    print("Mean Absolute Error: {}".format(mae))
    print("R2 Score: {}".format(r2))
printscores()    

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

    

