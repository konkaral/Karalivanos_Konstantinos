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
