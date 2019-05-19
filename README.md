# Karalivanos_Konstantinos

The following structure is followed in the code:

EDA
Basic satistics on numerical variables
Data Quality
Visual Analysis
Correlations
Feature Engineering and Selection
Feature Creation
Feature Selection
Machine Learning Models
Linear Reg Trial on Original Dataset hour_df
Random Forest

The data description is as follows:

Original columns:
weathersit: 1: Clear, Few clouds, Partly cloudy, 2: Mist and Cloudy, Mist and Broken clouds, Mist and Few clouds, Mist 3: Light Snow, Light Rain and Thunderstorm and Scattered clouds, Light Rain an dScattered clouds 4: Heavy Rain and Ice Pallets and Thunderstorm and Mist, Snow and Fog instant: record index

dteday: date

season: season (1:spring, 2:summer, 3:fall, 4:winter)

yr: year (0: 2011, 1:2012)

mnth: month ( 1 to 12)

holiday: weather day is holiday or not (extracted from http://dchr.dc.gov/page/holiday-schedule)

weekday: day of the week

workingday: if day is neither weekend nor holiday is 1, otherwise is 0.

temp: Normalized temperature in Celsius. The values are divided to 41 (max)

atemp: Normalized feeling temperature in Celsius. The values are divided to 50 (max)

hum: Normalized humidity. The values are divided to 100 (max)

windspeed: Normalized wind speed. The values are divided to 67 (max)

casual: count of casual users


registered: count of registered users

cnt: count of total rental bikes including both casual and registered




