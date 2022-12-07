import pickle

from django.db import models

# Create your models here.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
import numpy as np
from sklearn import model_selection, preprocessing
from sklearn.model_selection import cross_val_predict, cross_val_score, cross_validate, train_test_split
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from flask import Flask, render_template, request, jsonify
from sklearn.ensemble import RandomForestRegressor

data =  pd.read_csv("SeoulBikeData.csv", encoding="latin-1")

data.rename(columns = {"Rented Bike Count":"Rented_bike_count","Temperature(°C)" : "Temperature", "Humidity(%)": "Humidity", "Wind speed (m/s)": "Wind_speed", "Visibility (10m)":"Visibility", "Dew point temperature(°C)": "Dew_point_temperature", "Solar Radiation (MJ/m2)": "Solar_Radiation", "Rainfall(mm)": "Rainfall", "Snowfall (cm)": "Snowfall"}, inplace=True)

data['Hour'] = data['Hour'].astype(str)
data=data.drop(columns="Date")
data=pd.get_dummies(data,sparse=True)

X= data.drop(columns = 'Rented_bike_count')
y = data["Rented_bike_count"]
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2,random_state=42)

from sklearn.ensemble import RandomForestRegressor

rf_model=RandomForestRegressor()
rf_model.fit(X_train,y_train)

y_pred=rf_model.predict(X_test)


def ScaleData(x_train, x_test):
    scaler = preprocessing.StandardScaler(with_mean=False)
    scaler.fit(x_train)
    x_train = scaler.transform(x_train, copy=False)
    x_test = scaler.transform(x_test, copy=False)

ScaleData(X_train, X_test)

rf_model=RandomForestRegressor()
rf_model.fit(X_train,y_train)

pickle.dump(rf_model, open('model.pkl', "wb"))


