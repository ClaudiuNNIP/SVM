import quandl
import numpy as np 
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import os
import sklearn as sk

df = pd.read_excel(r"C:\Users\XM97SG\Onedrive - NN\Documents\Support Vector Regression\Python\close_price_wig20.xlsx", index_col=0, parse_dates=True)

df = df[np.isfinite(df['Trade_Close'])]

df = df[np.isfinite(df['Trade_Volume'])]

df.drop(['Trade_Volume'], axis=1)

df = pd.DataFrame(df['Trade_Close'])

# df['Returns'] = np.log(df / df.shift())

df.dropna(inplace=True)

#df = df.drop(['Trade_Close'], axis=1)

print(df)

# A variable for predicting 'n' days out into the future

forecast_out = 30 #'n=30' days

#Create another column (the target ) shifted 'n' units up

df['Prediction'] = df[['Trade_Close']].shift(-forecast_out)

#print the new data set

print(df.tail())


### Create the independent data set (X)  #######
# Convert the dataframe to a numpy array
X = np.array(df.drop(['Prediction'],1))

#Remove the last '30' rows
X = X[:-forecast_out]
print(X)


### Create the dependent data set (y)  #####
# Convert the dataframe to a numpy array 
y = np.array(df['Prediction'])
# Get all of the y values except the last '30' rows
y = y[:-forecast_out]
print(y)


# Split the data into 80% training and 20% testing
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# Create and train the Support Vector Machine (Regressor) 
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1) 
svr_rbf.fit(x_train, y_train)


# Testing Model: Score returns the coefficient of determination R^2 of the prediction. 
# The best possible score is 1.0
svr_confidence = svr_rbf.score(x_test, y_test)
print("svr confidence: ", svr_confidence)

# Create and train the Linear Regression  Model
lr = LinearRegression()
# Train the model
lr.fit(x_train, y_train)

# Testing Model: Score returns the coefficient of determination R^2 of the prediction. 
# The best possible score is 1.0
lr_confidence = lr.score(x_test, y_test)
print("lr confidence: ", lr_confidence)



x_forecast = np.array(df.drop(['Prediction'],1))[-forecast_out:]
print(x_forecast)

lr_prediction = lr.predict(x_forecast)
print(lr_prediction)

svm_prediction = svr_rbf.predict(x_forecast)
print(svm_prediction)
