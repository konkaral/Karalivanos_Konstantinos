# This is the dask implementation on the bike sharing dataset

import pandas as pd

# Setting the seed for reproducability

import random
random.seed(21)

# Import libraries 

import missingno as msno
import dask.dataframe as dd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from dask_ml.preprocessing import DummyEncoder

# Load the hourly dataset

hour_df = dd.read_csv("/Users/konstantinoskaralivanos/Desktop/hour.csv")

# Drop the column instance as the instance will not provide any information

hour_df = hour_df.drop(
    ["instant"], axis=1
)  

# Take a first look at the variables

hour_df.head()  
hour_df.info()
hour_df.head()

# Renaming columns names to more readable names

hour_df.columns = [
    "datetime",
    "season",
    "year",
    "month",
    "hour",
    "is_holiday",
    "weekday",
    "is_workingday",
    "weathersit",
    "temp",
    "atemp",
    "humidity",
    "windspeed",
    "casual",
    "registered",
    "cnt",
]


# Setting proper data types

# date time conversion

hour_df["datetime"] = dd.to_datetime(hour_df.datetime)

# categorical variables

hour_df["is_holiday"] = hour_df.is_holiday.astype("category")
hour_df["weekday"] = hour_df.weekday.astype("category")
hour_df["weathersit"] = hour_df.weathersit.astype("category")
hour_df["is_workingday"] = hour_df.is_workingday.astype("category")
hour_df["year"] = hour_df.year.astype("category")
hour_df["hour"] = hour_df.hour.astype("category")

plt.figure(figsize=(20, 5))
mask = np.zeros_like(hour_df.corr(), dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

fig, (ax1, ax2) = plt.subplots(ncols=2)
sns.boxplot(data=hour_df[["cnt", "casual", "registered"]].compute(), ax=ax1)
sns.boxplot(data=hour_df[["temp", "windspeed", "humidity"]].compute(), ax=ax2)
sns.heatmap(hour_df.corr(), cmap="RdBu_r", mask=mask, annot=True)

fig, ax = plt.subplots()
sns.boxplot(data=hour_df[["cnt", "hour"]].compute(), x="hour", y="cnt", ax=ax)
ax.set(title="Checking for outliers in day hours")

# Lets have a look at the IQR for the variable "windspeed"

from scipy import stats

Q1 = hour_df["windspeed"].quantile(0.25)
Q3 = hour_df["windspeed"].quantile(0.75)
IQR = Q3 - Q1
print(IQR)

#Using the IQR we will remove outliers

hour_df_out = hour_df[
    ~(
        (hour_df["windspeed"] < (Q1 - 3.6 * IQR))
        | (hour_df["windspeed"] > (Q3 + 3.6 * IQR))
    )
]
hour_df_out.shape

# Next have a look if there is a seasonal component in our dataset. 
# As a fact, usage peeks during summer and fall, which makes sense. (for casual users)

sns.set()
plt.figure(figsize=(11, 5))
sns.barplot(
    "year", "casual", hue="season", data=hour_df.compute(), palette="rainbow", ci=None
)
plt.legend(loc="upper right", bbox_to_anchor=(1.2, 0.5))
plt.xlabel("Year")
plt.ylabel("Total number of bikes rented on Casual basis")
plt.title("Number of bikes rented per season")

sns.set()
plt.figure(figsize=(11, 5))
sns.barplot(
    "year",
    "registered",
    hue="season",
    data=hour_df.compute(),
    palette="rainbow",
    ci=None,
)
plt.legend(loc="upper right", bbox_to_anchor=(1.2, 0.5))
plt.xlabel("Year")
plt.ylabel("Total number of bikes rented on Registered basis")
plt.title("Number of bikes rented per season")

# Seasonal distribution of rentals (sum)

f, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(13, 6))

ax1 = (
    hour_df_out[["season", "cnt"]]
    .compute()
    .groupby(["season"])
    .sum()
    .reset_index()
    .plot(
        kind="bar",
        legend=False,
        title="Counts of Bike Rentals by season",
        stacked=True,
        fontsize=12,
        ax=ax1,
    )
)
ax1.set_xlabel("season", fontsize=12)
ax1.set_ylabel("Count", fontsize=12)
ax1.set_xticklabels(["Spring", "Summer", "Fall", "Winter"])

ax2 = (
    hour_df_out[["weathersit", "cnt"]]
    .compute()
    .groupby(["weathersit"])
    .sum()
    .reset_index()
    .plot(
        kind="bar",
        legend=False,
        stacked=True,
        title="Counts of Bike Rentals by weathersit",
        fontsize=12,
        ax=ax2,
    )
)

ax2.set_xlabel("weathersit", fontsize=12)
ax2.set_ylabel("Count", fontsize=12)
ax2.set_xticklabels(["1: Clear", "2: Mist", "3: Light Snow", "4: Heavy Rain"])

f.tight_layout()

# Splitting for whether the day is a workday or not gives us useful insights.

sns.barplot(x="month", y="cnt", hue="is_workingday", data=hour_df_out.compute())
plt.show()

# Additionally check how the year influences the average and see that the average usage in year 2012 was always higher.

sns.barplot(x="month", y="cnt", hue="year", data=hour_df_out.compute())
plt.show()

# Hourly distribution of rentals over month (average)

# Configuring plotting visual and sizes
sns.set_style("whitegrid")
sns.set_context("talk")
params = {
    "legend.fontsize": "x-large",
    "figure.figsize": (30, 10),
    "axes.labelsize": "x-large",
    "axes.titlesize": "x-large",
    "xtick.labelsize": "x-large",
    "ytick.labelsize": "x-large",
}

plt.rcParams.update(params)

fig, ax = plt.subplots()
sns.pointplot(
    data=hour_df_out[["hour", "cnt", "season"]].compute(),
    x="hour",
    y="cnt",
    hue="season",
    ax=ax,
)
ax.set(title="Season wise hourly distribution of counts")

# Daily distribution of rentals (average)

sns.barplot("hour", "cnt", data=hour_df_out.compute(), ci=None)

# Weekday distribution of rentals (average)

sns.set()
sns.barplot("weekday", "cnt", data=hour_df_out.compute(), ci=None)

# Hourly distribution of rentals per weekdays (average)

# Configuring plotting visual and sizes
sns.set_style("whitegrid")
sns.set_context("talk")
params = {
    "legend.fontsize": "x-large",
    "figure.figsize": (30, 10),
    "axes.labelsize": "x-large",
    "axes.titlesize": "x-large",
    "xtick.labelsize": "x-large",
    "ytick.labelsize": "x-large",
}

plt.rcParams.update(params)
fig, ax = plt.subplots()
sns.pointplot(
    data=hour_df_out[["hour", "cnt", "weekday"]].compute(),
    x="hour",
    y="cnt",
    hue="weekday",
    ax=ax,
)
ax.set(title="Season wise hourly distribution of counts")

# Average distribution whether it is holiday or not

sns.set()
sns.barplot("is_holiday", "cnt", data=hour_df_out.compute(), ci=None)

# Average distribution whether workday or not

sns.set()
sns.barplot("is_workingday", "cnt", data=hour_df_out.compute(), ci=None)

# To understand how the variables are correlated to the target variable visual representations are developed

plt.figure(figsize=(20, 5))
mask = np.zeros_like(hour_df_out.compute().corr(), dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(hour_df.corr(), cmap="RdBu_r", mask=mask, annot=True)

# Weather conditions

sns.set()
sns.lineplot("atemp", "cnt", data=hour_df.compute(), palette="rainbow", ci=None)

# Windspeed

sns.set()
sns.lineplot("windspeed", "cnt", data=hour_df.compute(), palette="rainbow", ci=None)

sns.set()
sns.lineplot("windspeed", "cnt", data=hour_df_out.compute(), palette="rainbow", ci=None)

# Weathersit

sns.set()
sns.lineplot("weathersit", "cnt", data=hour_df.compute(), palette="rainbow", ci=None)

# Feature Engineering

# Calculate the real temperature using the above equation

hour_df_out["temp_real"] = 47 * hour_df_out["temp"] - 8

# Using the results from real temperature in Celcius, I can use these values to calculate the heat index or humiture (HI), 
# which is an index that combines air temperature and relative humidity. The formula used for calculation is as follows: 
# HI = c1 + c2T + c3R + c4TR + c5T^2 + c6R^2 + c7RT^2 + c8TR^2 + c9T^2R^2 
# where T is the temperature (in degrees Celcius) and R is the relative humidity (percentage value between 0 and 100)
# c1 = −8.78469475556, c2 = 1.61139411, c3 = 2.33854883889, c4 = -0.14611605, 
# c5 = -0.012308094, c6 = -0.0164248277778, c7 = 0.002211732, c8 = 0.00072546, c9 = -0.000003582

hour_df_out["heat_index"] = (
    -8.78469475556
    + (1.61139411 * (hour_df_out["temp_real"]))
    + (2.33854883889 * hour_df_out["humidity"] * 100)
    + (-0.14611605 * hour_df_out["temp_real"] * hour_df_out["humidity"] * 100)
    + (-0.012308094 * (hour_df_out["temp_real"]) ** 2)
    + (-0.0164248277778 * (hour_df_out["humidity"] * 100) ** 2)
    + (
        0.002211732
        * ((hour_df_out["temp_real"]) ** 2)
        * (hour_df_out["humidity"] * 100)
    )
    + (0.00072546 * hour_df_out["temp_real"] * (hour_df_out["humidity"] * 100) ** 2)
    + (
        -0.000003582
        * ((hour_df_out["humidity"] * 100) * 2)
        * ((hour_df_out["temp_real"]) * 2)
    )
)

# Wind Chill Index (WCI)

# I can also calculate the Wind Chill Index (WCI) which is the lowering of body temperature due to the passing-flow of 
# lower-temperature air. The formula used is as follows: 
# WCI = (10SQRT(windspeed) - windspeed + 10.5)(33 - temp_real)
# where:
# WCI = wind chill index, kcal/m2/h v = wind velocity, m/s Ta = air temperature, °C
# since the wind speed is in km/h we will use a 0.277778 coefficient to convert it to m/s

hour_df_out["WCI"] = (
    10 * np.sqrt(hour_df_out["windspeed"] * 0.277778 * 100)
    - (0.277778 * hour_df_out["windspeed"])
    + 10.5
) * (33 - hour_df_out["temp_real"])

# Humidity Index (HX)

# The humidex (“humidity Index”, abbreviated to HX in the present study) is a measure of the combined effect of heat and 
# humidity on human physiology. It is calculated from air temperature and relative humidity. 
# First, the vapour pressure of water v (in hPa) is calculated using:
# v = (6.112 × 10ˆ(7.5*T/(237.7 + T)) * RH/100)
# where T = air temperature (°C) and RH is the relative humidity (%).
# The Humidex (HX) is then found using: HX = T + (v − 10) * 5 / 9

hour_df_out["v"] = (
    6.112 * 10 ** (7.5 * hour_df_out["temp_real"] / (237.7 + hour_df_out["temp_real"]))
) * (hour_df_out["humidity"])

hour_df_out["humidex"] = hour_df_out["temp_real"] + ((hour_df_out["v"] - 10) * (5 / 9))

# check correlations again

# Based on this we identified multicollinearity between season and month and decided to drop month as season 
# has a higher correlation with the target variable cnt

plt.figure(figsize=(20, 5))
mask = np.zeros_like(hour_df_out.corr(), dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(hour_df_out.corr(), cmap="RdBu_r", annot=True)

# Feature Selection

# temp

# temp vs Target

plt.scatter(hour_df_out[["temp"]].compute(), hour_df_out[["cnt"]].compute(), alpha=0.5)
plt.title("temp Vs Target('cnt')")
plt.xlabel("temp")
plt.ylabel("Bike sharing count")
plt.show()

# atemp

# atemp vs Target

plt.scatter(hour_df_out[["atemp"]].compute(), hour_df_out[["cnt"]].compute(), alpha=0.5)
plt.title("atemp Vs Target('cnt')")
plt.xlabel("atemp")
plt.ylabel("Bike sharing count")
plt.show()

# humidity 

# temp vs Target

plt.scatter(
    hour_df_out[["humidity"]].compute(), hour_df_out[["cnt"]].compute(), alpha=0.5
)
plt.title("humidity Vs Target('cnt')")
plt.xlabel("humidity")
plt.ylabel("Bike sharing count")
plt.show()

# windspeed 

# windspeed vs Target

plt.scatter(
    hour_df_out[["windspeed"]].compute(), hour_df_out[["cnt"]].compute(), alpha=0.5
)
plt.title("windspeed Vs Target('cnt')")
plt.xlabel("windspeed")
plt.ylabel("Bike sharing count")
plt.show()

# casual

# casual vs Target

plt.scatter(
    hour_df_out[["casual"]].compute(), hour_df_out[["cnt"]].compute(), alpha=0.5
)
plt.title("casual Vs Target('cnt')")
plt.xlabel("casual")
plt.ylabel("Bike sharing count")
plt.show()

# registered 

# registered vs Target

plt.scatter(
    hour_df_out[["registered"]].compute(), hour_df_out[["cnt"]].compute(), alpha=0.5
)
plt.title("registered Vs Target('cnt')")
plt.xlabel("registered")
plt.ylabel("Bike sharing count")
plt.show()

# temp_real

# temp_real vs Target

plt.scatter(
    hour_df_out[["temp_real"]].compute(), hour_df_out[["cnt"]].compute(), alpha=0.5
)
plt.title("temp_real Vs Target('cnt')")
plt.xlabel("temp_real")
plt.ylabel("Bike sharing count")
plt.show()

# heat index

# heat_index vs Target

plt.scatter(
    hour_df_out[["heat_index"]].compute(), hour_df_out[["cnt"]].compute(), alpha=0.5
)
plt.title("heat_index Vs Target('cnt')")
plt.xlabel("heat_index")
plt.ylabel("Bike sharing count")
plt.show()

# WCI 

# WCI vs Target

plt.scatter(hour_df_out[["WCI"]].compute(), hour_df_out[["cnt"]].compute(), alpha=0.5)
plt.title("WCI Vs Target('cnt')")
plt.xlabel("WCI")
plt.ylabel("Bike sharing count")
plt.show()

# v

# WCI vs Target

plt.scatter(hour_df_out[["v"]].compute(), hour_df_out[["cnt"]].compute(), alpha=0.5)
plt.title("v Vs Target('cnt')")
plt.xlabel("v")
plt.ylabel("Bike sharing count")
plt.show()

# humidex

# WCI vs Target
plt.scatter(
    hour_df_out[["humidex"]].compute(), hour_df_out[["cnt"]].compute(), alpha=0.5
)
plt.title("humidex Vs Target('cnt')")
plt.xlabel("humidex")
plt.ylabel("Bike sharing count")
plt.show()

# Modelling

# import libraries

from sklearn import linear_model
import math
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score
import datetime as dt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from dask_ml.linear_model import LinearRegression
from dask_ml.metrics import r2_score

# linear regression

lr_model = LinearRegression()

traintrial = hour_df[hour_df["datetime"].compute() < "2012-10-01"]
testtrial = hour_df[hour_df["datetime"].compute() >= "2012-10-01"]

# drop the target variable and "registered" variable since we are using registered along with the rest of features

X_traintrial = traintrial.drop(
    ["cnt", "registered", "month", "casual", "season"], axis=1
)
y_traintrial = traintrial["cnt"]
X_testtrial = testtrial.drop(["cnt", "registered", "month", "casual"], axis=1)
y_testtrial = testtrial["cnt"]
X_traintrial = X_traintrial.drop(["datetime"], axis=1)
X_testtrial = X_testtrial.drop(["datetime"], axis=1)

# for DummyEncoder to work need to categorize the variables first

X_traintrial = X_traintrial.categorize(
    ["year", "hour", "is_holiday", "weekday", "is_workingday", "weathersit"]
)

X_testtrial = X_testtrial.categorize(
    ["year", "hour", "is_holiday", "weekday", "is_workingday", "weathersit"]
    
de = DummyEncoder(
    ["year", "hour", "is_holiday", "weekday", "is_workingday", "weathersit"]
)
X_traintrial = de.fit_transform(X_traintrial)


de = DummyEncoder(
    ["year", "hour", "is_holiday", "weekday", "is_workingday", "weathersit"]
)
X_testtrial = de.fit_transform(X_testtrial)
)

lr = LinearRegression()
lr.fit(X_traintrial.values, y_traintrial.values)

X_testtrial = X_testtrial.drop("season", axis=1)

y_predtrial = lr.predict(X_traintrial.values)

y_predtest = lr.predict(X_testtrial.values)

import dask.array as da

y_predtest.compute()

# random forest 

rf = RandomForestRegressor(n_estimators=1000, max_depth=10)

rf.fit(X_traintrial, y_traintrial)

rf_pred = rf.predict(X_testtrial)

# in case a value is predicted as minus

rf_pred = [i if i >= 0 else 0 for i in rf_pred]  

# round prediction count to the nearest integer

rf_pred = [round(x) for x in rf_pred]

# root mean squared error

print("RMSE: %.2f" % np.sqrt(mean_squared_error(y_testtrial, rf_pred)))

# feature importance

feature_importance = rf.feature_importances_
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + 0.5
plt.figure(figsize=(12, 10))
plt.barh(pos, feature_importance[sorted_idx], align="center")
plt.yticks(pos, X_traintrial.columns[sorted_idx])
plt.xlabel("Relative Importance")
plt.title("Variable Importance")
plt.show()
