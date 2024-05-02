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
    train = False
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
    target_feature = 'TOTALDEMAND'
    input_features = [  # 25
        'TOTALDEMAND',  # continuous
        'FORECASTDEMAND',  # continuous
        'TEMPERATURE',  # continuous
        'rrp',  # continuous
        'daily_avg_actual',  # continuous
        'daily_avg_forecast',  # continuous
        'forecast_error',  # continuous
        'smoothed_forecast_demand',  # continuous
        'year',  # cyclical
        'quarter',  # cyclical
        'month',  # cyclical
        'week_of_year',  # cyclical
        'dow',  # cyclical
        'doy',  # cyclical
        'day_of_month',  # cyclical
        'hour',  # cyclical
        'is_weekend',  # boolean, categorical
        'part_of_day',  # categorical
        'season',  # categorical
        'is_business_day',  # boolean, categorical
        'smoothed_total_demand',  # continuous
        'smoothed_temperature',  # continuous
        'minutes_past_midnight',  # cyclical
        'season_name'  # categorical
    ]
    continuous_features = [  # 10
        'TOTALDEMAND',
        'FORECASTDEMAND',
        'TEMPERATURE',
        'rrp',
        'daily_avg_actual',
        'daily_avg_forecast',
        'forecast_error',
        'smoothed_forecast_demand',
        'smoothed_total_demand',
        'smoothed_temperature'
    ]
    cyclical_features = [  # 9
        'hour',
        'dow',
        'doy',
        'month',
        'quarter',
        'week_of_year',
        'minutes_past_midnight'
    ]
    categorical_features = [  # 5
        'is_weekend',
        'part_of_day',
        'season',
        'is_business_day',
        'season_name'
    ]
    max_values = {  # define the maximum values for each cyclical feature
        'year': 2021,
        'quarter': 4,
        'month': 12,
        'week_of_year': 52,
        'dow': 7,
        'doy': 365,  # todo: 366 for leap years to be more precise
        'day_of_month': 31,
        'hour': 24,
        'minutes_past_midnight': 1439
    }
    # column_mapping = {
    #     'TOTALDEMAND': 'normalised_total_demand',
    #     'FORECASTDEMAND': 'normalised_forecast_demand',
    #     'TEMPERATURE': 'normalised_temperature',
    #     'rrp': 'normalised_rrp',
    #     'forecast_error': 'normalised_forecast_error',
    #     'smoothed_forecast_demand': 'normalised_smoothed_forecast_demand',
    #     'smoothed_total_demand': 'normalised_smoothed_total_demand',
    #     'smoothed_temperature': 'normalised_smoothed_temperature',
    # }
