"""

"""
from xgb_functions import *
import wandb
import gc

n_in = 6  # 6 lag intervals
n_test = 336  # 7 days of 30-minute sample intervals


# Objective function to be optimized
def objective():
    """
    Objective function for hyperparameter tuning.
    :param config: Configuration for hyperparameters (using W&B API).
    :return:
    """
    # initialize a W&B run
    wandb.init()

    # create a trial to suggest hyperparameters
    trial = optuna.trial.FixedTrial({
        'gamma': wandb.config.gamma,
        'learning_rate': wandb.config.learning_rate,
        'max_depth': wandb.config.max_depth,
        'min_child_weight': wandb.config.min_child_weight,
        'subsample': wandb.config.subsample,
        'reg_alpha': wandb.config.reg_alpha,
        'reg_lambda': wandb.config.reg_lambda,
        'device': 'cuda',
        'tree_method': 'hist',
    })

    # define the hyperparameters to be tuned
    params = {
        'objective': 'reg:squarederror',
        'gamma': trial.suggest_float('gamma', 0.1, 6.0),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 300),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0),
        'device': 'cuda',
        'tree_method': 'hist'
    }

    df = pd.read_csv('../data/NSW/final_df.csv', index_col=0)
    data = series_to_supervised(df, n_in=n_in)
    X, y = data.iloc[:, :-1], data.iloc[:, -1]
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.1,
        shuffle=False
    )

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    # Train the model
    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        evals=[(dtrain, 'train'), (dval, 'eval')],
        early_stopping_rounds=50,
        verbose_eval=False
    )

    # Predict on validation data
    y_pred = bst.predict(dval)
    error = mean_absolute_error(y_val, y_pred)

    # Log the metric
    wandb.log({'mae': error})

    wandb.finish()
    gc.collect()


if __name__ == "__main__":
    # define the sweep configuration
    sweep_config = {
        'method': 'bayes',  # Can be grid, random, bayes
        'metric': {
            'name': 'mae',
            'goal': 'minimize'
        },
        'parameters': {
            'gamma': {'min': 0.1, 'max': 6.0},
            'learning_rate': {'min': 0.001, 'max': 0.3},
            'max_depth': {'min': 3, 'max': 10},
            'min_child_weight': {'min': 1, 'max': 300},
            'subsample': {'min': 0.5, 'max': 1.0},
            'reg_alpha': {'min': 0.0, 'max': 1.0},
            'reg_lambda': {'min': 1e-8, 'max': 10.0},
            # add any other parameters you wish to tune
        }
    }

    # initialize a W&B sweep
    sweep_id = wandb.sweep(sweep_config, project='electricity_demand_forecasting')

    # run the W&B sweep
    wandb.agent(sweep_id, objective, count=100)
