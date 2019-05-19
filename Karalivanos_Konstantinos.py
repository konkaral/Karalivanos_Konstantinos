#This is the dask implementation on the bike sharing dataset.
import pandas as pd
# setting the seed for reproducability
import random
random.seed(21)
import missingno as msno
import dask.dataframe as dd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from dask_ml.preprocessing import DummyEncoder
# load the hourly dataset
hour_df = dd.read_csv("/Users/konstantinoskaralivanos/Desktop/hour.csv")
hour_df = hour_df.drop(
    ["instant"], axis=1
)  # drop the column instance as the instance will not provide any information

hour_df.head(5)  # take a first look at the variables
