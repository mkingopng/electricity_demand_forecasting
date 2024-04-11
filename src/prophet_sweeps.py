"""

"""
import pandas as pd
import wandb
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from sklearn.metrics import mean_absolute_error


class CFG:
    wandb_project = 'electricity_demand_forecasting'
    data_path = '../data/NSW/nsw_df.parquet'
    sweep_config = {
        'method': 'random',  # alternately grid or bayes
        'metric': {
            'name': 'mae',
            'goal': 'minimize'
        },
        'parameters': {
            'changepoint_prior_scale': {
                'min': 0.001,
                'max': 0.5
            },
            'seasonality_prior_scale': {
                'min': 1.0,
                'max': 10.0
            },
            'daily_fourier_order': {
                'min': 3,
                'max': 20
            },
            'weekly_fourier_order': {
                'min': 3,
                'max': 20
            },
            'monthly_fourier_order': {
                'min': 3,
                'max': 20
            },
            'quarterly_fourier_order': {
                'min': 3,
                'max': 20
            }
        }
    }


def train():
    with wandb.init(project=CFG.wandb_project, config=CFG.sweep_config) as run:
        config = run.config
        df = pd.read_parquet(CFG.data_path)
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'ds', 'TOTALDEMAND': 'y'}, inplace=True)
        df['ds'] = pd.to_datetime(df['ds'])

        # test-train split
        cutoff_date = df['ds'].max() - pd.Timedelta(days=7)
        df_train = df[df['ds'] <= cutoff_date]
        df_test = df[df['ds'] > cutoff_date]

        model = Prophet(
            changepoint_prior_scale=config.changepoint_prior_scale,
            seasonality_prior_scale=config.seasonality_prior_scale,
            weekly_seasonality=False)

        model.fit(df_train)

        # Perform cross-validation
        df_cv = cross_validation(
            model,
            initial='8762 hours',  # 365 days
            period='2160 hours',  # 90 days
            horizon='168 hours'  # 7 days
        )

        df_p = performance_metrics(df_cv)

        for _, row in df_p.iterrows():
            wandb.log({
                'horizon': row['horizon'].total_seconds() / 3600,
                'mae': row['mae'],
                'rmse': row['rmse'],
                'mape': row['mape'],
            })

        # Directly use df_test to create 'future' df for predictions
        future = df_test[[
            'ds',
            'TEMPERATURE',
            'FORECASTDEMAND',
            'rrp',
            'smoothed_total_demand',
            'smoothed_temperature',
            'smoothed_forecast_demand',
            'year',
            'quarter',
            'month',
            'dow',
            'doy',
            'hour',
            'season'
        ]].copy()

        forecast = model.predict(future)

        # calc and log MAE or other performance metrics using actual y
        # values from df_val and predicted yhat from forecast
        mae = mean_absolute_error(df_test['y'], forecast['yhat'])
        wandb.log({'mae': mae})


# initialize the sweep
sweep_id = wandb.sweep(
    sweep=CFG.sweep_config,
    project=CFG.wandb_project
)

# start sweep
wandb.agent(sweep_id, train, count=100)  # adj count as necessary
