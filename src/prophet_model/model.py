"""
model component for Prophet model
"""
from prophet import Prophet
from prophet.diagnostics import cross_validation
from sklearn.metrics import mean_absolute_error
import os
from prophet.serialize import model_to_json, model_from_json
import wandb


class ProphetModel:
    """
    Prophet model class
    """
    def __init__(self, config):
        self.config = config
        self.model = None

    def initialize_model(self):
        """
        initialize the prophet model
        """
        self.model = Prophet(weekly_seasonality=False)
        # add all regressors
        regressors = [
            'TEMPERATURE', 'FORECASTDEMAND', 'rrp', 'smoothed_total_demand',
            'smoothed_temperature', 'smoothed_forecast_demand', 'year',
            'quarter', 'month', 'dow', 'doy', 'hour', 'season'
        ]
        for reg in regressors:
            self.model.add_regressor(reg)

        # add seasonality
        self.model.add_seasonality(name='daily', period=48, fourier_order=6)
        self.model.add_seasonality(name='weekly', period=336, fourier_order=3)
        self.model.add_seasonality(name='monthly', period=1464, fourier_order=5)
        self.model.add_seasonality(name='quarterly', period=4380, fourier_order=3)

        # add holidays
        self.model.add_country_holidays(country_name='AU')

    def train(self, df_train, df_test):
        """
        train the model
        """
        self.initialize_model()
        self.model.fit(df_train)
        df_cv = self.cross_validate(df_test)  # cross-validate using test set
        return df_cv

    def cross_validate(self):
        """
        perform cross-validation
        """
        df_cv = cross_validation(
            self.model,
            initial='8762 hours',
            period='2160 hours',
            horizon='168 hours'
        )
        return df_cv

    def load_model(self):
        with open(os.path.join(self.config.trained_models, 'trained_prophet_model.json'), 'r') as fin:
            self.model = model_from_json(fin.read())

    def predict(self, df_val):
        """
        make predictions
        """
        forecast = self.model.predict(df_val)
        return forecast

    def evaluate(self, df_val, forecast):
        """
        evaluate the model using MAE
        """
        val_mae = mean_absolute_error(df_val['y'], forecast['yhat'])
        wandb.log({'Validation MAE': val_mae})
        return val_mae

    def save_model(self):
        """
        save the trained model
        """
        with open(os.path.join(self.config.trained_models, 'trained_prophet_model_1.json'), 'w') as fout:
            fout.write(model_to_json(self.model))
