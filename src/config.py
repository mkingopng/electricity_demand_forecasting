"""

"""


class CFG:
    data_path = './../data/NSW'
    img_dim1 = 20
    img_dim2 = 10
    n_in = 6  # 6 lag intervals
    n_test = 336  # 7 days of 30-minute sample intervals
    wandb_project_name = 'electricity_demand_forecasting'
    wandb_run_name = 'xgboost'
    version = 13  # increment for each new experiment
    logging = True  # set to True to enable W&B logging
    sweep_count = 10  # number of sweep runs
    params = {
        'objective': 'reg:squarederror',
        'gamma': 4.592513457496951,  # def 0
        'learning_rate': 0.07984076257805875,  # def 0.1
        'max_depth': 8,
        'min_child_weight': 20,  # def 0.1
        'nthread': 4,  # ?
        'random_state': 42,
        'reg_alpha': 0.7863437272577511,
        'reg_lambda': 3.475149811652308,  # def 1
        'eval_metric': ['mae'],
        'tree_method': 'hist'
    }
    sweep_config = {
        "method": "random",
        "parameters": {
            "learning_rate": {
                "min": 0.001,
                "max": 1.0
            },
            "gamma": {
                "min": 0.001,
                "max": 1.0
            },
            "min_child_weight": {
                "min": 1,
                "max": 150
            },
            "early_stopping_rounds": {
                "values": [10, 20, 30, 40]
            },
        }
    }


class LstmCFG:
    n_splits = 5
    n_features = 35
    input_size = 1  # the number of input features in dataset
    hidden_layer_size = 50
    output_size = 1
    learning_rate = 0.001
    batch_size = 1
    epochs = 10
    sequence_length = 336  # one week of 30-minute sample intervals
