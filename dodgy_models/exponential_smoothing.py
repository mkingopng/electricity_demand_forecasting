import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import numpy as np
from dotenv import load_dotenv
from statsmodels.tsa.api import ExponentialSmoothing

load_dotenv()
wandb_api_key = os.getenv('WANDB_API_KEY')
wandb.login(key=wandb_api_key)  # log-in to wandb using API key


class CFG:
    data_path = '../data'
    img_dim1 = 20
    img_dim2 = 10
    wandb_project = 'electricity_demand_forecasting'  # Set your project name
    wandb_run_name = 'simple_exponential_smoothing'  # Name this run


# initialize wandb
wandb.init(
    project=CFG.wandb_project,
    name=CFG.wandb_run_name
)

# settings
sns.set(style="darkgrid")

df = pd.read_csv(
    os.path.join(CFG.data_path, 'NSW', 'final_df.csv'),
    index_col=0
)

#########
# train-test-validation split

# select features & target variable
X = df[['TEMPERATURE']]  # we can modify this based on feature analysis
y = df['TOTALDEMAND']

# Determine the indices for splitting
train_size = int(len(X) * 0.6)
validation_size = int(len(X) * 0.2)
test_size = len(X) - train_size - validation_size

# Split the datasets
X_train, X_temp = X[:train_size], X[train_size:]
y_train, y_temp = y[:train_size], y[train_size:]

X_val, X_test = X_temp[:validation_size], X_temp[validation_size:]
y_val, y_test = y_temp[:validation_size], y_temp[validation_size:]

###################################
# Exponential smoothing
# Convert index to datetime, if not already
df.index = pd.to_datetime(df.index)

# Manually set the frequency to 30 minutes when converting to a PeriodIndex
df.index = df.index.to_period(freq='30min')

# Assuming 'df' is your DataFrame and 'y' is the target variable
y = df['TOTALDEMAND']

# Fit the model
model = ExponentialSmoothing(y, seasonal_periods=48, trend='add', seasonal='add', use_boxcox=True, initialization_method="estimated")
fit = model.fit()

# Forecast the next 24 periods (adjust according to your needs)
forecast = fit.forecast(24)

# Plotting the results
plt.figure(figsize=(CFG.img_dim1, CFG.img_dim2))
plt.plot(y.reset_index(drop=True), label='Actual Demand')  # Reset index for plotting
# Plot forecast with an offset for continuation
plt.plot(range(len(y), len(y)+len(forecast)), forecast, label='Forecast', linestyle='--')

# Optionally, set custom date labels for the x-axis
date_labels = y.index.to_timestamp().strftime('%YYYY')[::48]  # Adjust the slicing based on your needs
plt.xticks(ticks=range(0, len(y), 48), labels=date_labels, rotation=45)  # Adjust ticks for your dataset

plt.title('Demand Forecast using Holt-Winters’ Seasonal Method')
plt.xlabel('Time')
plt.ylabel('Electricity Demand')
plt.legend()
plt.tight_layout()  # Adjust layout to make room for date labels
plt.show()

# fix_me: Save the plot to a file
forecast_plot_filename = "exponential_smoothing_forecast.png"
plt.savefig(forecast_plot_filename)
plt.close()

# Log the saved plot to wandb
wandb.log({"Exponential Smoothing Forecast": wandb.Image(forecast_plot_filename)})

y_test = df['TOTALDEMAND'][-24:]  # adjust according to dataset & analysis
forecast = fit.forecast(24)

mae = mean_absolute_error(y_test, forecast)
mse = mean_squared_error(y_test, forecast)
rmse = np.sqrt(mse)
mape = mean_absolute_percentage_error(y_test, forecast)

# log metrics to wandb
wandb.log({
    'MAE': mae,
    'MSE': mse,
    'RMSE': rmse,
    'MAPE': mape
})

##########################################

# finish wandb run
wandb.finish()
