import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Data Processing
df = pd.read_csv('2015-08-15_Austin_Sustainability_Indicators.csv') # data frame
x = df.iloc[:, :-1] # columns 1-4 (Date, Chart Date, Topic Area, Key Performance Indicator
y = df.iloc[:, -1] # value column from data set


performance_indicator_array = [] # empty array to get performance metrics

for i in x['Key Performance Indicator'].unique(): # get unique metrics
    performance_indicator_array += [i] # append unique metric to list

performance_indicators = df[['Date', 'Key Performance Indicator', 'Value']]

bike_mi_df = performance_indicators[performance_indicators['Key Performance Indicator'] == 'Miles of new and improved bike facilities']
bike_mi_df = bike_mi_df.sort_values(by='Date')
bike_mi_df['Date'] = pd.DatetimeIndex(bike_mi_df['Date']).year
bike_mi_df['Date'] = bike_mi_df['Date']

x = bike_mi_df.iloc[:, 0].values
y = bike_mi_df.iloc[:, 2].values

# Visualization of New Miles overtime
plt.bar(x, y, color='blue')
plt.xlabel('Year')
plt.ylabel('Miles')
plt.title('City of Austin: New Bike Miles per Year')
plt.show()