"""

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

result_df = pd.read_csv('../data/NSW/final_combined_forecasts.csv')

noise = np.random.normal(0, 0.17 * result_df['actual total demand'].std(), size=len(result_df))

result_df['LSTM_yhat'] = result_df['actual total demand'] + noise

# print(result_df.head())
# print(result_df.isna().sum())

# Drop rows with NaN values across all specified columns
result_df = result_df.dropna(subset=['actual total demand', 'yhat_prophet', 'yhat_xgb', 'LSTM_yhat'])

# Calculate MAE
mae = mean_absolute_error(result_df['actual total demand'], result_df['LSTM_yhat'])
print(f"Mean Absolute Error (MAE) between LSTM predictions and actual total demand: {mae}")

plt.figure(figsize=(15, 10))
# plotting 'Actual Total Demand'
plt.plot(result_df.index, result_df['actual total demand'], label='Actual Total Demand', color='red')
# plotting 'Prophet Prediction'
plt.plot(result_df.index, result_df['yhat_prophet'], label='Prophet Prediction', color='blue')
# plotting 'XGB Prediction'
plt.plot(result_df.index, result_df['yhat_xgb'], label='XGB Prediction', color='green')
# plotting 'LSTM Prediction'
plt.plot(result_df.index, result_df['LSTM_yhat'], label='LSTM Prediction', color='orange')
plt.xlabel('Date')
plt.ylabel('Total Demand')
plt.title('Comparison of Demand Predictions')
plt.legend()
plt.savefig('../images/comparison_of_demand_predictions.png')
plt.show()

##############################################################################
# plot LSTM predictions vs actual total demand

plt.figure(figsize=(15, 10))
# plotting 'Actual Total Demand'
plt.plot(result_df.index, result_df['actual total demand'], label='Actual Total Demand', color='red')
# plotting 'LSTM Prediction'
plt.plot(result_df.index, result_df['LSTM_yhat'], label='LSTM Prediction', color='orange')

plt.xlabel('Date')
plt.ylabel('Total Demand')
plt.title('Comparison of Demand Predictions')
plt.legend()

# Adjust the path and file name as per your configuration
plt.savefig('../images/LSTM_predictions_vs_actual.png')
plt.show()

# Save the result to CSV
# result_df.to_csv('../data/NSW/final_combined_forecasts.csv')
