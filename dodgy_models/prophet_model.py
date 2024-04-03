"""
trying to get prophet to work in this instance
"""
import pandas as pd
import itertools
import numpy as np
from random import gauss

import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

from pandas.plotting import autocorrelation_plot
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import RFECV
from sklearn.linear_model import Ridge

import warnings
import itertools
from pandas.plotting import autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from dateutil.easter import easter
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_yearly, add_changepoints_to_plot

import matplotlib.pyplot as plt

# Load and prepare data
df = pd.read_csv('./../data/NSW/processed_data.csv')
df.rename(columns={'Unnamed: 0': 'ds', 'TOTALDEMAND': 'y'}, inplace=True)

# ensure 'ds' is datetime type
df['ds'] = pd.to_datetime(df['ds'])

print(df.head())

# initial plot of the data
plt.figure(figsize=(15, 10))
plt.plot(df['ds'], df['y'], label='Total Demand')
plt.xlabel('Date')
plt.ylabel('Total Demand')
plt.title('Total Demand Over Time')
plt.legend()
plt.show()


m = Prophet()
m.fit(df)

# make future dataframe
future = m.make_future_dataframe(periods=30)  # next 30 days

# forecast
forecast = m.predict(future)

# plot forecast
fig = m.plot(forecast)
add_changepoints_to_plot(fig.gca(), m, forecast)
plt.show()

# plot components of the forecast
fig2 = m.plot_components(forecast)
plt.show()
