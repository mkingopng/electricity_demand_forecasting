"""
main.py
"""
from .config import CFG
from .dataloader import DataLoader
from .model import ProphetModel
from .visualisation import Visualization
import wandb


# setup configuration, data loader, model, visualizer
config = CFG()
data_loader = DataLoader(config)
model = ProphetModel(config)
visualizer = Visualization(config)

# Initialize Weights & Biases
wandb.init(
    project=CFG.wandb_project,
    name=CFG.wandb_run_name
)


def main(train_mode=True):
    # load and split data
    df = data_loader.load_data()
    df_train, df_test, df_val = data_loader.split_data(df)

    if train_mode:
        df_cv = model.train(df_train, df_test)
        model.save_model()
        forecast = model.predict(df_val)
        val_mae = model.evaluate(df_val, forecast)
        print(f"Validation MAE: {val_mae}")

        # plotting inside training
        visualizer.plot_total_demand_over_time(df)
        visualizer.plot_cross_validation_metric(df_cv)
        visualizer.plot_forecast_vs_actual(forecast, df_val)
        visualizer.plot_components(model.model, forecast)
    else:
        model.load_model()
        return df_val


# Use this for prediction mode visualizations
def plot_predictions(df_val):
    # prepare the features for prediction
    future = df_val[[
        'ds', 'TEMPERATURE', 'FORECASTDEMAND', 'rrp',
        'smoothed_total_demand', 'smoothed_temperature',
        'smoothed_forecast_demand', 'year', 'quarter', 'month', 'dow',
        'doy', 'hour', 'season']].copy()
    forecast = model.predict(future)
    # print(forecast['yhat'])
    # print(df_val['ds'], df_val['y'])
    return forecast


if __name__ == "__main__":
    # decide the mode based on the configuration
    df_val = main(train_mode=CFG.train)

    if not CFG.train:
        forecast = plot_predictions(df_val)

        # call visualization methods
        visualizer.plot_total_demand_over_time(df_val)
        visualizer.plot_forecast_vs_actual(forecast, df_val)
        visualizer.plot_change_points(model.model, forecast)
        visualizer.plot_components(model.model, forecast)

