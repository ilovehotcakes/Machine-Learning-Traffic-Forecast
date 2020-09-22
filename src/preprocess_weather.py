# This script processing weather data, data scource is from NOAA.gov
# 
# This script will interpolation the missing values.
#
# It contains 31 inputs and we are only keeping the one that is revelant
# The full list of names and explanation of the inputs can be found in 3505doc.txt
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt


# Read file from disk
filename = 'raw_data/weather/renton_010116_013120_weather_original.txt'
raw_file = open(filename, 'r')
raw_data = raw_file.read()
raw_file.close()
rows = raw_data.split('\n')  # Split data by row


# Chop all the data after 'PCP01' off
# PCP01 is percipitation per hour
for i in range(len(rows)):
    rows[i] = rows[i][0:126]


# Write raw_data to temporary file
with open('temp.csv','w') as temp_file:
    for item in rows:
        temp_file.write('%s\n' % item)


# Load temp file into pandas
df = pd.read_csv('temp.csv', delim_whitespace=True)


# Convert data into time series data
df['Datetime'] = pd.to_datetime(df['YR--MODAHRMN'], format='%Y%m%d%H%M')  # Formatting YR--MODAHRMN to pd datetime
df.set_index('Datetime', inplace=True)                                    # Set Date as the index column


# Remove unused data columns
del df['USAF']
del df['WBAN']
del df['YR--MODAHRMN']
del df['DIR']
del df['GUS']
del df['SKC']
del df['L']
del df['M']
del df['H']
del df['MW']
del df['MW.1']
del df['MW.2']
del df['MW.3']
del df['AW']
del df['AW.1']
del df['AW.2']
del df['AW.3']
del df['W']
del df['MAX']
del df['MIN']


# Replace the missing values with -1
df.loc[df['SPD'] == '***', 'SPD'] = -1
df.loc[df['CLG'] == '***', 'CLG'] = -1
df.loc[df['VSB'] == '****', 'VSB'] = -1
df.loc[df['TEMP'] == '****', 'TEMP'] = -1
df.loc[df['DEWP'] == '****', 'DEWP'] = -1
df.loc[df['SLP'] == '******', 'SLP'] = -1
df.loc[df['ALT'] == '*****', 'ALT'] = -1
df.loc[df['STP'] == '******', 'STP'] = -1
df.loc[df['PCP01'] == '*****', 'PCP01'] = -1
# print(df)


# Convert type from string to int
df['SPD'] = pd.to_numeric(df['SPD'])
df['CLG'] = pd.to_numeric(df['CLG'])
df['VSB'] = pd.to_numeric(df['VSB'])
df['TEMP'] = pd.to_numeric(df['TEMP'])
df['DEWP'] = pd.to_numeric(df['DEWP'])
df['SLP'] = pd.to_numeric(df['SLP'])
df['ALT'] = pd.to_numeric(df['ALT'])
df['STP'] = pd.to_numeric(df['STP'])
df['PCP01'] = pd.to_numeric(df['PCP01'])
print(df)
df['PCP01'].plot()  # Visually checking the data, anything that is -1 is missing data
plt.show()


# Resample the data so the data is every hour on the hour
# I.e. 2020-01-31 23:53:00 will become 2020-01-31 23:00:00
newDF = df.resample('H').max()  # Choose max values for each hour


# Change all the -1 values (missin values) to NaN
newDF = newDF.replace(to_replace=-1, value=np.nan)


# Interpolate missing values -> 1490 rows out of 35808 rows
newDF = newDF.interpolate(method ='linear', limit_direction ='forward').round(2)


# Write the processed data to disk
newDF.to_csv('processed_data/renton_010116_013120_weather.csv', sep=',', encoding='utf-8')


# Remove temp file after finish processing
os.remove('temp.csv')
