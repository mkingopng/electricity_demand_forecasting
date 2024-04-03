import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import numpy as np
from dotenv import load_dotenv

load_dotenv()
wandb_api_key = os.getenv('WANDB_API_KEY')
wandb.login(key=wandb_api_key)  # log-in to wandb using API key


class CFG:
    data_path = '../data'
    img_dim1 = 20
    img_dim2 = 10
    wandb_project = 'electricity_demand_forecasting'  # Set your project name
    wandb_run_name = 'baseline_linear_regression'  # Name this run


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

# todo: replace this reuben's function
from src.utils import subset_data



###########
# linear regression

# initialise model
model = LinearRegression()

# fit model
model.fit(X_train, y_train)

# make predictions
y_pred = model.predict(X_test)

# calculate error metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)

# Log metrics to wandb
wandb.log({
    'MSE': mse,
    'RMSE': rmse,
    'MAE': mae,
    'MAPE': mape
})

# print the error metrics
print(f'MSE: {mse}')
print(f'RMSE: {rmse}')
print(f'MAE: {mae}')
print(f'MAPE: {mape}')

# plot the actual vs predicted values
plt.figure(figsize=(CFG.img_dim1, CFG.img_dim2))
plt.plot(y_test.values, label='Actual Demand', color='blue', alpha=0.6)
plt.plot(y_pred, label='Predicted Demand', color='red', alpha=0.6)
plt.title('Actual vs Predicted Electricity Demand')
plt.xlabel('Time')
plt.ylabel('Electricity Demand')
plt.legend()

# save plot to wandb
plt.savefig("actual_vs_predicted.png")
wandb.log({"Actual vs Predicted Electricity Demand": wandb.Image("actual_vs_predicted.png")})
plt.show()

# feature importance
feature_importance = pd.Series(index=X_train.columns, data=np.abs(model.coef_))
feature_importance.sort_values().plot(kind='barh', figsize=(CFG.img_dim1, CFG.img_dim2))
plt.title('Feature Importance')

# save feature importance plot to wandb
plt.savefig("feature_importance.png")
wandb.log({"Feature Importance": wandb.Image("feature_importance.png")})
plt.show()

##########################################

# finish wandb run
wandb.finish()
