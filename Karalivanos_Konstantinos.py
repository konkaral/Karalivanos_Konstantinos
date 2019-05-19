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
