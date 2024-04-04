"""

"""
import xgboost as xgb
import optuna
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

n_in = 6  # 6 lag intervals
n_test = 336  # 7 days of 30-minute sample intervals

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True, target_var='TOTALDEMAND'):
    """
    frame a time series dataset as a supervised learning dataset, required as
    in input for xgb
    :param data:
    :param n_in:
    :param n_out:
    :param dropnan:
    :param target_var:
    :return:
    """
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()

    for i in range(n_in, 0, -1):  # input sequence (t-n, ... t-1)
        cols.append(df.drop(columns=target_var).shift(i))
        names += [('%s(t-%d)' % (df.columns[j], i)) for j in range(n_vars) if
                  df.columns[j] != target_var]

    # forecast sequence (t+1, ... t+n_out-1), only for predictors, not target
    for i in range(1, n_out):
        cols.append(df.drop(columns=target_var).shift(-i))
        names += [('%s(t+%d)' % (df.columns[j], i)) for j in range(n_vars) if
                  df.columns[j] != target_var]

    # add the target variable column at t (current timestep)
    cols.append(df[[target_var]])
    names.append('%s(t)' % target_var)

    # combine everything
    agg = pd.concat(cols, axis=1)
    agg.columns = names

    # drop records with NaN values
    if dropnan:
        agg.dropna(inplace=True)

    return agg


def objective(trial):
    # define the hyperparameters to be tuned
    params = {
        'objective': 'reg:squarederror',
        'gamma': trial.suggest_float('gamma', 0.1, 6.0),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 300),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        # 'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
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

    # train the model
    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        evals=[(dtrain, 'train'), (dval, 'eval')],
        early_stopping_rounds=50,
        verbose_eval=False  # turn off verbose output
    )

    # predict on validation data
    y_pred = bst.predict(dval)
    error = mean_absolute_error(y_val, y_pred)

    # return the metric you want to minimize
    return error


if __name__ == "__main__":

    # start the Optuna optimization
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=4000)

    # best trial details
    print(study.best_trial)

"""
FrozenTrial(
number=75, 
state=1, 
values=[49.21131582650541], 
datetime_start=datetime.datetime(2024, 4, 4, 21, 34, 15, 648534), 
datetime_complete=datetime.datetime(2024, 4, 4, 21, 34, 33, 319116), 
params={
'gamma': 4.592513457496951, 
'learning_rate': 0.07984076257805875, 
'max_depth': 8, 
'min_child_weight': 20, 
'subsample': 0.9383519651035563, 
'reg_alpha': 0.7863437272577511, 
'reg_lambda': 3.475149811652308
}, 
user_attrs={}, system_attrs={}, intermediate_values={}, distributions={'gamma': FloatDistribution(high=5.0, log=False, low=0.1, step=None), 'learning_rate': FloatDistribution(high=0.3, log=False, low=0.01, step=None), 'max_depth': IntDistribution(high=10, log=False, low=3, step=1), 'min_child_weight': IntDistribution(high=300, log=False, low=1, step=1), 'subsample': FloatDistribution(high=1.0, log=False, low=0.5, step=None), 'reg_alpha': FloatDistribution(high=1.0, log=False, low=0.0, step=None), 'reg_lambda': FloatDistribution(high=10.0, log=False, low=1e-08, step=None)}, trial_id=75, value=None)

"""

# todo:
#  CUDA device issue,
#  improve data,
#  save trained model,
#  ensemble,
#  feature importance,
#  decomposition features
#  visualisation using SHAP
