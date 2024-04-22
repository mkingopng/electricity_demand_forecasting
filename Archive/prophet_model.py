"""
Prophet is a univariate additive time series model, which supports trends,
seasonality, and holidays
"""
import pandas as pd
from sklearn.metrics import mean_absolute_error
import warnings
from prophet import Prophet
from prophet.plot import add_changepoints_to_plot
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_cross_validation_metric
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import wandb
from prophet.serialize import model_to_json, model_from_json
import os

warnings.simplefilter(action='ignore', category=FutureWarning)
load_dotenv()


class CFG:
	wandb_project = 'electricity_demand_forecasting'
	wandb_run_name = 'prophet'
	data_path = '../data/NSW'
	image_path = '../images'
	trained_models = '../trained_models'
	train = False


wandb.init(
	project=CFG.wandb_project,
	name=CFG.wandb_run_name,
)

# load and prep data
df = pd.read_parquet(os.path.join(CFG.data_path, 'nsw_df.parquet'))

# reset index to make subsequent changes
df.reset_index(inplace=True)

# prophet has some eccentricities for naming features
df.rename(
	columns={'index': 'ds', 'TOTALDEMAND': 'y'},
	inplace=True
)

# ensure 'ds' is datetime type
df['ds'] = pd.to_datetime(df['ds'])

# print(df['TEMPERATURE'].isna().sum())  # debugging line

# train - test - validation split
# calculate the cut-off date for the last 7 days (val_df)
cutoff_date = df['ds'].max() - pd.Timedelta(days=7)

# calculate the cut-off date for the last 14 days (test_df)
cutoff_date2 = df['ds'].max() - pd.Timedelta(days=14)

# create training df by slicing data at the cut-off date
df_train = df[df['ds'] <= cutoff_date2]

# create testing df the same way
df_test = df[(df['ds'] > cutoff_date2) & (df['ds'] <= cutoff_date)]

# validation df
df_val = df[df['ds'] > cutoff_date]

# print(df.head())

# initial plot of the data
plt.figure(figsize=(15, 10))
plt.plot(df['ds'], df['y'], label='Total Demand')
plt.xlabel('Date')
plt.ylabel('Total Demand')
plt.title('Total Demand Over Time')
plt.legend()
plt.savefig(os.path.join(CFG.image_path, 'total_demand_over_time.png'))
plt.show()

##############################################################################

if CFG.train:  # if script is in training mode
	# initialise the model
	model = Prophet(weekly_seasonality=False)

	# add regressors
	model.add_regressor('TEMPERATURE')
	model.add_regressor('FORECASTDEMAND')
	model.add_regressor('rrp')
	model.add_regressor('smoothed_total_demand')
	model.add_regressor('smoothed_temperature')
	model.add_regressor('smoothed_forecast_demand')
	model.add_regressor('year')
	model.add_regressor('quarter')
	model.add_regressor('month')
	model.add_regressor('dow')
	model.add_regressor('doy')
	model.add_regressor('hour')
	model.add_regressor('season')

	# add daily seasonality
	model.add_seasonality(
		name='daily',
		period=48,  # 48 x 30 minute samples per day
		fourier_order=6  # selected based on sweeps
	)

	# add weekly seasonality
	model.add_seasonality(
		name='weekly',
		period=336,  # 7 days of 30 minute sample periods
		fourier_order=3  # selected based on sweeps
	)

	# add monthly seasonality (approximately 30.5 days in a month)
	model.add_seasonality(
		name='monthly',
		period=1464,
		fourier_order=5  # selected based on sweeps
	)

	# quarterly seasonality
	model.add_seasonality(
		name='quarterly',
		period=4380,
		fourier_order=3  # selected based on sweeps
	)

	# add country holidays
	model.add_country_holidays(country_name='AU')

	model.fit(df_train)

	# cross validation
	df_cv = cross_validation(
		model,
		initial='8762 hours',  # 365 days
		period='2160 hours',  # 90 days
		horizon='168 hours'  # 7 days
	)

	# prediction and performance metrics
	train_forecast = model.predict(df_train)
	train_mae = mean_absolute_error(df_train['y'], train_forecast['yhat'])
	print(f"Training MAE: {train_mae}")
	wandb.log({'Training MAE': train_mae})

	forecast = model.predict(df_val)
	test_mae = mean_absolute_error(df_val['y'], forecast['yhat'])
	print(f"Testing MAE: {test_mae}")
	wandb.log({'Testing MAE': test_mae})

	fig = plot_cross_validation_metric(
		df_cv,
		metric='mae'
	)
	plt.savefig(os.path.join(CFG.image_path, 'prophet_cv_mae.png'))
	plt.show()

	print(f"Overall Training MAE: {train_mae}")
	print(f"Overall Testing MAE: {test_mae}")

	with open('../trained_models/trained_prophet_model.json', 'w') as fout:
		fout.write(model_to_json(model))  # Save model
############################################################################
else:  # if script is in prediction mode
	with open(os.path.join(CFG.trained_models, 'trained_prophet_model.json')) as fin:
		model = model_from_json(fin.read())  # Load model
		# make future dataframe
		future = df_val[[
			'ds',
			'TEMPERATURE',
			'FORECASTDEMAND',
			'rrp',
			'smoothed_total_demand',
			'smoothed_temperature',
			'smoothed_forecast_demand',
			'year',
			'quarter',
			'month',
			'dow',
			'doy',
			'hour',
			'season'
		]].copy()

	# forecast
	forecast = model.predict(future)
	print(forecast['yhat'])
	print(df_val['ds'], df_val['y'])

	# save actual and forecast values to a new dataframe
	new_df = pd.merge(df_val[['ds', 'y']], forecast[['ds', 'yhat']], on='ds')
	new_df.to_csv(os.path.join(CFG.data_path, 'prophet_forecasts.csv'))

	print(new_df.head())

	# plot change points
	fig1 = model.plot(forecast)
	add_changepoints_to_plot(fig1.gca(), model, forecast)
	plt.savefig(os.path.join(CFG.image_path, 'prophet_changepoints.png'))
	plt.show()

	# plot forecast and actual values for test period
	plt.figure(figsize=(10, 6))
	# plot forecast
	plt.plot(forecast['ds'], forecast['yhat'], label='Forecast', color='blue')
	# plot actual values
	plt.plot(df_val['ds'], df_val['y'], label='Actual', color='red')
	plt.xlabel('Date')
	plt.ylabel('Total Demand')
	plt.title('Forecast vs Actual for the Test Period')
	plt.legend()
	plt.savefig(os.path.join(CFG.image_path, 'prophet_forecast_vs_actual.png'))
	plt.show()

	# plot components of the forecast
	fig2 = model.plot_components(forecast)
	plt.savefig(os.path.join(CFG.image_path, 'prophet_components.png'))
	plt.show()

	# check forecast df only includes dates present in the test set
	forecast_filtered = forecast[forecast['ds'].isin(df_val['ds'])]

	# calculate MAE
	mae = mean_absolute_error(df_val['y'], forecast_filtered['yhat'])
	wandb.log({'mae': mae})
	wandb.finish()

	print(f"Mean Absolute Error (MAE): {mae}")
