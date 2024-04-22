"""

"""
import matplotlib.pyplot as plt
from prophet.plot import plot_cross_validation_metric
from prophet.plot import add_changepoints_to_plot
import os
from config import CFG


class Visualization:
    def __init__(self, config):
        self.config = config

    def plot_data(self, df):
        plt.figure(figsize=(CFG.img_dim1, CFG.img_dim2))
        plt.plot(df['ds'], df['y'], label='Total Demand')
        plt.xlabel('Date')
        plt.ylabel('Total Demand')
        plt.title('Total Demand Over Time')
        plt.legend()
        plt.savefig(os.path.join(self.config.image_path, 'total_demand_over_time.png'))
        plt.show()

    def plot_total_demand_over_time(self, df):
        """
        Plot the total demand over time (raw data)
        """
        plt.figure(figsize=(CFG.img_dim1, CFG.img_dim2))
        plt.plot(df['ds'], df['y'], label='Total Demand')
        plt.xlabel('Date')
        plt.ylabel('Total Demand')
        plt.title('Total Demand Over Time')
        plt.legend()
        plt.savefig(os.path.join(self.config.image_path, 'total_demand_over_time.png'))
        plt.show()

    def plot_forecast(self, forecast, actual):
        # plotting logic
        plt.show()

    def plot_cross_validation_metric(self, df_cv):
        """
        Plot cross-validation metric
        """
        fig = plot_cross_validation_metric(df_cv, metric='mae')
        plt.savefig(os.path.join(self.config.image_path, 'prophet_cv_mae.png'))
        plt.show()

    def plot_change_points(self, model, forecast):
            fig = model.plot(forecast)
            add_changepoints_to_plot(fig.gca(), model, forecast)
            plt.savefig(os.path.join(self.config.image_path, 'prophet_changepoints.png'))
            plt.show()

    def plot_forecast_vs_actual(self, forecast, df_val):
        plt.figure(figsize=(CFG.img_dim1, CFG.img_dim2))
        plt.plot(forecast['ds'], forecast['yhat'], label='Forecast', color='blue')
        plt.plot(df_val['ds'], df_val['y'], label='Actual', color='red')
        plt.xlabel('Date')
        plt.ylabel('Total Demand')
        plt.title('Forecast vs Actual for the Test Period')
        plt.legend()
        plt.savefig(os.path.join(self.config.image_path, 'prophet_forecast_vs_actual.png'))
        plt.show()

    def plot_components(self, model, forecast):
        fig = model.plot_components(forecast)
        plt.savefig(os.path.join(self.config.image_path, 'prophet_components.png'))
        plt.show()
