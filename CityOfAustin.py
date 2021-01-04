import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Data Processing
df = pd.read_csv('2015-08-15_Austin_Sustainability_Indicators.csv') # data frame
x = df.iloc[:, :-1] # columns 1-4 (Date, Chart Date, Topic Area, Key Performance Indicator
y = df.iloc[:, -1] # value column from data set


performance_indicator_array = [] # empty array to get performance metrics

for i in x['Key Performance Indicator'].unique(): # get unique metrics
    print(i)
    performance_indicator_array += [i] # append unique metric to list

performance_indicators = df[['Date', 'Key Performance Indicator', 'Value']]

# ************************************** Bike Miles Created ****************************************************
bike_mi_df = performance_indicators[performance_indicators['Key Performance Indicator'] == 'Miles of new and improved bike facilities']
bike_mi_df = bike_mi_df.sort_values(by='Date')
bike_mi_df['Date'] = pd.DatetimeIndex(bike_mi_df['Date']).year
bike_mi_df['Date'] = bike_mi_df['Date']

years = mdates.YearLocator()


x = bike_mi_df.iloc[:, 0].values
y = bike_mi_df.iloc[:, 2].values

# Visualization of New Miles Overtime
#plt.xticks(x)
#plt.plot(x, y, color='blue')
#plt.xlabel('Year')
#plt.ylabel('Miles')
#plt.title('City of Austin: New Bike Miles per Year')
# plt.show()

# ******************* Number of Traffic Signals Retimed ***********************************************
traffic_sig_df = performance_indicators[performance_indicators['Key Performance Indicator'] ==
                                        'Miles of new and improved bike facilities']
traffic_sig_df = bike_mi_df.sort_values(by='Date')

x1 = traffic_sig_df.iloc[:, 0].values
y1 = traffic_sig_df.iloc[:, -1].values
plt.xticks(x1)
barchart = plt.bar(x1, y1, color='blue')
plt.xlabel('Year')
plt.ylabel('Number of Traffic Signals Retimed')
plt.title('City of Austin: Annual number of Retimed Signals')
plt.show()

# ******************* CO2 Emissions - Community-wide (mtC02e) ***********************************************

# co2_df = performance_indicators[performance_indicators['Key Performance Indicator'] == 'CO2 Emissions - Community-wide (mtCO2e)']
# co2_df = co2_df.sort_values(by='Date')
# x2 = co2_df.iloc[:, 0].values
# y2 = co2_df.iloc[:, -1].values
# plt.xticks(x2)
# plt.plot(x2, y2, color='red')
# plt.xlabel('Year')
# plt.ylabel('CO2 Emissions (MegaTons)')
# plt.title('City of Austin Emissions over Time')
# plt.show()




