# This script processes the traffic data and combines it with the processed weather
# data and outputs a csv file for training
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing


# Reading .csv file from disk using pandas
# Traffic file contains Date, DayOfWeek, Hour, and 20 speed bins (e.g. 0-5, 5-10, ..., 90-95 mph)
file_sb = 'raw_data/traffic/d1_010116_013120_speed_southbound.csv'  # Southbound traffic data
file_nb = 'raw_data/traffic/d1_010116_013120_speed_northbound.csv'  # Northbound traffic data
traffic = pd.read_csv(file_sb)


# Converting to pandas time series
# Change hour from 1-24 to 0-23 so pandas can convert to time series
traffic['Hour'] = traffic['Hour'] - 1
# Adding 0 paddings for Hour so pandas can understand
traffic['Hour_Padded'] = traffic['Hour'].apply(lambda x: '{0:0>2}'.format(x))
# Formatting 'Date' and 'Hour' to pd datetime
datatime = traffic['Date'].map(str) + ' ' + traffic['Hour_Padded']
# Convert to time series
traffic['Date'] = pd.to_datetime(datatime)
# Set Date as the index column
traffic.set_index('Date', inplace=True)
# Hour_Padded is in str format, Hour is in int format, we are keeping the int format
del traffic['Hour_Padded']
print(traffic, '\n')  # Checking data


# Aggregating some of the data that is missing using the existing data
# Counting total traffic per hour
traffic['Total'] = traffic.sum(axis=1)           # Total traffic for each hour
# Convert DayOfWeek from name to numerical value 
traffic['DayOfWeek'] = traffic.index.weekday     # 0 is Monday and 6 is Sunday
# Convert day of year name to numerical value
traffic['DayOfYear'] = traffic.index.dayofyear   # 1 is the Jan 1st
print(traffic['2016-01-17':'2019-01-23'], '\n')  # A slice of data from Jan 2016

# Calculate average traffic speed for each hour, the formula is:
# Total speed per hour / total traffic per hour
#  = (sum of (total traffic per bin per hour * mean speed per bin per hour)) / total traffic per hour
traffic['AvgSpeed'] = 0  # New col for avg speed
for j in range (19):     # 19 speed bins
    column = str(j * 5) + '-' + str((j + 1) * 5)    # Bin column name, e.g. 5-10
    speed = j * 5 + 2.5                             # Avg speed of a bin, e.g. 5-10 bin's avg is 7.5
    traffic['AvgSpeed'] += traffic[column] * speed  # Sum up the total speed
traffic['AvgSpeed'] /= traffic['Total']             # Total speed per hour / total traffic per hour
# print(traffic['2019-03-29':'2019-03-30']['AvgSpeed'], '\n')  # Checking data


# Scaling data - Not used; data is scaler in training file so it can be unscaled
# traffic[:] = preprocessing.scale(traffic[:])
# print(traffic['2019-01':'2019-02'])  # Checking data


# Joining the processed traffic data with the processed weather data
weather = pd.read_csv('processed_data/renton_010116_013120_weather.csv')
weather.set_index('Datetime', inplace=True)  # Set Datetime as index
result = traffic.join(weather, how='outer')  # Combine both sets of data
print(result)
# result.to_csv('processed_data/all_data_sb.csv', sep=',', encoding='utf-8')  # Write to disk
# result.to_csv('processed_data/all_data_nb.csv', sep=',', encoding='utf-8')  # Write to disk


# Visualizing average speed data
plt.figure(figsize=(12,5))
plt.xlabel('Datetime')
plt.ylabel('AvgSpeed')
traffic['2016-01-17':'2016-01-23']['AvgSpeed'].plot()
# traffic['AvgSpeed'].plot()

# Spot checking data 
# AvgSpeed2019 = traffic['2019-03-24':'2019-03-31']['AvgSpeed']
# AvgSpeed2019.plot()
# AvgSpeed2018 = traffic['2018-03-24':'2018-03-31']['AvgSpeed']
# AvgSpeed2018.plot()

plt.show()
