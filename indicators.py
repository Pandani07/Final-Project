from preprocessing import getdataset
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import random
import pickle

df50 = getdataset()


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

x1_train, x1_test, y1_train, y1_test = train_test_split(x, y, test_size = 0.2, random_state=0)
classifier.fit(x1_train, y1_train.values.ravel())
y_pred = classifier.predict(x1_test)

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
print("Accuracy of the Classifier: {}\n".format(accuracy_score(y1_test, y_pred)))
print("Confusion Matrix:\n")
cm = confusion_matrix(y1_test, y_pred)
print(cm)
print("\n")
print("Classification Report:\n")
cr = classification_report(y1_test, y_pred)
print(cr)

pkl_filename = "pickle_model.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(classifier, file)
    



