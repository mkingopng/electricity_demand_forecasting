"""

"""
import matplotlib.pyplot as plt
from prophet.plot import plot_cross_validation_metric
from prophet.plot import add_changepoints_to_plot
import os
from .config import CFG


class Visualization:
    def __init__(self, config):
        self.config = config

    def plot_data(self, df):
        """
        plot the total demand over time (raw data)
        """
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
        plot the total demand over time (raw data)
        isn't this a duplicate of the previous method?
        """
        plt.figure(figsize=(CFG.img_dim1, CFG.img_dim2))
        plt.plot(df['ds'], df['y'], label='Total Demand')
        plt.xlabel('Date')
        plt.ylabel('Total Demand')
        plt.title('Total Demand Over Time')
        plt.legend()
        plt.savefig(os.path.join(self.config.image_path, 'total_demand_over_time.png'))
        plt.show()

    def plot_cross_validation_metric(self, df_cv):
        """
        plot cross-validation metric
        """
        fig = plot_cross_validation_metric(df_cv, metric='mae')
        plt.savefig(os.path.join(self.config.image_path, 'prophet_cv_mae.png'))
        plt.show()

    def plot_change_points(self, model, forecast):
        """
        plot the changepoints
        """
        fig = model.plot(forecast)
        add_changepoints_to_plot(fig.gca(), model, forecast)
        plt.savefig(
            os.path.join(
                self.config.image_path,
                'prophet_changepoints.png'
            )
        )
        plt.show()

    def plot_forecast_vs_actual(self, forecast, df_val):
        """
        plot the forecast vs actual values
        """
        plt.figure(figsize=(CFG.img_dim1, CFG.img_dim2))
        # plot the forecast values
        plt.plot(
            forecast['ds'],
            forecast['yhat'],
            label='Forecast',
            color='green'
        )
        # plot the actual values
        plt.plot(
            df_val['ds'],
            df_val['y'],
            label='Actual',
            color='red'
        )
        # other plot components
        plt.xlabel('Date')
        plt.ylabel('Total Demand')
        plt.title('Forecast vs Actual for the Test Period')
        plt.legend()
        # save the plot
        plt.savefig(
            os.path.join(
                self.config.image_path,
                'prophet_forecast_vs_actual.png'
            )
        )
        plt.show()

    def plot_components(self, model, forecast):
        """
        Plot the components of the forecast
        """
        fig = model.plot_components(forecast)
        plt.savefig(os.path.join(self.config.image_path, 'prophet_components.png'))
        plt.show()
