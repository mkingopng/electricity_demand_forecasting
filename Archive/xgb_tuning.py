"""

"""
import pandas as pd
import xgboost as xgb
import optuna
from sklearn.metrics import mean_absolute_error
import gc
from xgb_config import CFG

n_in = 6  # 6 lag intervals
n_test = 336  # 7 days of 30-minute sample intervals


def series_to_supervised(data, n_in=1, dropnan=True, target_var='TOTALDEMAND'):
    df = pd.DataFrame(data)
    cols, names = [], []

    # Input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [(f"{col}(t-{i})") for col in df.columns]

    # Put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names

    # Add target
    agg[target_var] = df[target_var]

    # Drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)

    return agg


def objective(trial):
    params = {
        'objective': 'reg:squarederror',
        'gamma': trial.suggest_float('gamma', 0.1, 6.0),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 300),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0),
        'tree_method': 'hist'
    }

    df = pd.read_parquet('../data/NSW/nsw_df.parquet')
    df.drop(columns=['daily_avg_actual', 'daily_avg_forecast'], inplace=True)
    # df.index = pd.to_datetime(df.index)

    val_end_date = df.index.max()
    val_start_date = val_end_date - pd.Timedelta(days=7)
    test_end_date = val_start_date - pd.Timedelta(minutes=30)
    test_start_date = test_end_date - pd.Timedelta(days=7)

    df_train = df[df.index <= test_start_date]
    df_test = df[(df.index > test_start_date) & (df.index <= test_end_date)]
    df_val = df[df.index > val_start_date]

    train_supervised = series_to_supervised(df_train, n_in=n_in)
    test_supervised = series_to_supervised(df_test, n_in=CFG.n_in)
    val_supervised = series_to_supervised(df_val, n_in=n_in)

    n_obs = CFG.n_in * len(df.columns)

    trainX, trainy = train_supervised.iloc[:, :-1], train_supervised.iloc[:, -1]
    testX, testy = test_supervised.iloc[:, :-1], test_supervised.iloc[:, -1]
    valX, valy = val_supervised.iloc[:, :-1], val_supervised.iloc[:, -1]

    if testy.empty:
        raise ValueError("Validation set is empty. Check the date range or data preprocessing.")

    dtrain = xgb.DMatrix(trainX, label=trainy)
    dval = xgb.DMatrix(testX, label=testy)

    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        evals=[(dtrain, 'train'), (dval, 'eval')],
        early_stopping_rounds=50,
        verbose_eval=False
    )

    y_pred = bst.predict(dval)
    error = mean_absolute_error(testy, y_pred)
    gc.collect()
    return error


if __name__ == "__main__":
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=400)  # more trials
    print(study.best_trial)

# todo:
#  improve data,
#  save trained model,
#  feature importance,
#  visualisation using SHAP
