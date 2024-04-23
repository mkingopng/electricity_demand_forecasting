"""

"""
from config import CFG
from dataloader import DataLoader
from model import ProphetModel
from visualisation import Visualization
import wandb
import pandas as pd
import os


def main():
    # initialize Weights & Biases
    wandb.init(
        project=CFG.wandb_project,
        name=CFG.wandb_run_name
    )

    # setup configuration, data loader, model, visualiser
    config = CFG()
    data_loader = DataLoader(config)
    model = ProphetModel(config)
    visualiser = Visualization(config)

    # load and split data
    df = data_loader.load_data()
    df_train, df_test, df_val = data_loader.split_data(df)

    visualiser.plot_total_demand_over_time(df)

    # proceed with training or prediction based on configuration
    if config.train:
        df_cv = model.train(df_train, df_test)
        model.save_model()
        forecast = model.predict(df_val)
        val_mae = model.evaluate(df_val, forecast)
        print(f"Validation MAE: {val_mae}")

    else:
        model.load_model()

        # prepare the features for prediction
        future = df_val[[
            'ds', 'TEMPERATURE', 'FORECASTDEMAND', 'rrp',
            'smoothed_total_demand', 'smoothed_temperature',
            'smoothed_forecast_demand', 'year', 'quarter', 'month', 'dow',
            'doy', 'hour', 'season']].copy()
        forecast = model.predict(future)
        print(forecast['yhat'])
        print(df_val['ds'], df_val['y'])

        # save actual and forecast values to a new dataframe
        new_df = pd.merge(df_val[['ds', 'y']], forecast[['ds', 'yhat']], on='ds')
        new_df.to_csv(os.path.join(config.data_path, 'prophet_forecasts.csv'))
        print(new_df.head())

    wandb.finish()  # close wandb session


if __name__ == "__main__":
    main()
    config = CFG()
    data_loader = DataLoader(config)
    model = ProphetModel(config)
    visualiser = Visualization(config)

    if not CFG.train:
        visualiser.plot_change_points(model.model, forecast)
        visualiser.plot_forecast_vs_actual(forecast, df_val)
        visualiser.plot_components(model.model, forecast)


