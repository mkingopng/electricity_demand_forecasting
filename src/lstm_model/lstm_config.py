"""

"""

class LstmCFG:
    """
    configuration class for the LSTM model
    """
    wandb_project_name = 'electricity_demand_forecasting'
    wandb_run_name = 'lstm'
    data_path = '../../data/NSW'
    image_path = '../../images'
    logging = True
    version = 26
    n_folds = 10
    epochs = 30
    n_features = 33
    input_size = 33
    num_layers = 5
    hidden_units = 50
    output_size = 1
    lr = 0.0003
    batch_size = 1024
    seq_length = 336  # 336 one week of 30-minute sample intervals
    dropout = 0.2
    weight_decay = 0.00001
    lrs_step_size = 6
    lrs_gamma = 0.4
    train = True
    input_features = [
        'normalised_total_demand',
        'normalised_forecast_demand',
        'normalised_temperature',
        'normalised_rrp',
        'normalised_forecast_error',
        'normalised_smoothed_forecast_demand',
        'hour_sin',
        'hour_cos',
        'dow_sin',
        'dow_cos'
    ]
    max_values = {
        'hour': 24,
        'dow': 7,
        'doy': 365,  # todo: 366 for leap years to be more precise
        'month': 12,
        'quarter': 4
    }
    column_mapping = {
        'TOTALDEMAND': 'normalised_total_demand',
        'FORECASTDEMAND': 'normalised_forecast_demand',
        'TEMPERATURE': 'normalised_temperature',
        'rrp': 'normalised_rrp',
        'forecast_error': 'normalised_forecast_error',
        'smoothed_forecast_demand': 'normalised_smoothed_forecast_demand',
        'smoothed_total_demand': 'normalised_smoothed_total_demand',
        'smoothed_temperature': 'normalised_smoothed_temperature',
    }

