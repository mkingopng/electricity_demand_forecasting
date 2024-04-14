"""

"""


class LstmCFG:
    """
    configuration class for the LSTM model
    """
    wandb_project_name = 'electricity_demand_forecasting'
    wandb_run_name = 'lstm'
    data_path = './../data/NSW'
    image_path = './../images'
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
