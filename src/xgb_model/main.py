"""

"""
import pandas as pd
from data_processing import series_to_supervised, train_test_split, load_and_preprocess_data
from model import train_model, initialize_wandb, train_and_evaluate_model, save_model, load_model, evaluate_model
from evaluation import evaluate_predictions
from visualisation import (
    plot_shap_summary, plot_dependence_plots, plot_decision_plot,
    plot_waterfall_plot, plot_beeswarm_plot, plot_actual_vs_predicted,
    initialize_shap_explainer, plot_shap_values
)
from config import CFG
import os


def main():
    if CFG.train:
        # initialize W&B
        run = initialize_wandb()

        # load and preprocess the data
        dtrain, dtest, dval, trainy, testy, valy = load_and_preprocess_data()

        # train and evaluate the model
        model, mae = train_and_evaluate_model(dtrain, dtest, CFG.params)
        print(f"Mean Absolute Error on Test Set: {mae}")

        # save the model
        save_model(model)

        # finish W&B run
        if run:
            run.finish()
    else:
        # load the model
        model = load_model(os.path.join(CFG.models_path, 'xgb_model.json'))

        # load data necessary for evaluation
        _, _, dval, _, _, valy = load_and_preprocess_data()

        # evaluate the model
        val_mae, val_predictions = evaluate_model(model, dval, valy)
        print(f"Validation MAE: {val_mae}")

        # plots
        explainer, shap_values = initialize_shap_explainer(model, dval)
        plot_shap_summary(explainer, shap_values)
        plot_dependence_plots(explainer, shap_values, feature_index=0)  # example for the first feature
        plot_decision_plot(explainer, shap_values)
        plot_shap_values(explainer, dval, feature_to_exclude='FORECAST_DEMAND(t-1)')


if __name__ == "__main__":
    main()
