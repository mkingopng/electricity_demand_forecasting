"""

"""
from .data_processing import (
    series_to_supervised, train_test_split, load_and_preprocess_data
)
from .model import (
    train_model, initialize_wandb, train_and_evaluate_model, save_model,
    load_model, evaluate_model
)
from .visualisation import (
    plot_shap_summary, plot_dependence_plots, plot_decision_plot,
    plot_waterfall_plot, plot_beeswarm_plot, plot_actual_vs_predicted,
    initialize_shap_explainer, plot_shap_values
)
from .config import CFG
import os


# todo: correct setup as a package, add wandb logging, unit tests, exception handling
def main():
    if CFG.train:
        # initialize W&B
        run = initialize_wandb()

        # load and preprocess the data
        dtrain, dtest, dval, trainy, testy, valy, valX = load_and_preprocess_data()

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
        _, _, dval, _, _, valy, valX = load_and_preprocess_data()

        # Debug: Check the shape of valX right after loading
        # print("Shape of valX after loading:", valX.shape)

        # evaluate the model
        val_mae, val_predictions = evaluate_model(model, dval, valy)
        print(f"Validation MAE: {val_mae}")

        # todo: plots
        explainer, shap_values = initialize_shap_explainer(model, dval)
        # print("Shape of valX before plotting:", valX.shape)

        #
        plot_shap_summary(
            explainer,
            shap_values,
            image_name='SHAP summary.png'
        )

        #
        plot_dependence_plots(
            explainer,
            shap_values,
            feature_index=0,
            image_name="dependence_plot_feature_0.png"
        )

        #
        plot_dependence_plots(
            explainer,
            shap_values,
            feature_index=1,
            image_name='dependence_plot_feature_1.png'
        )

        #
        plot_decision_plot(
            explainer,
            valX,
            shap_values,
            instance_index=0,
            image_name='decision plot.png'
        )

        #
        plot_shap_values(
            explainer,
            valX,
            feature_to_exclude='FORECAST_DEMAND(t-1)',
            image_name='SHAP values excluding feature 0.png'
        )

        #
        # plot_beeswarm_plot(
        #     explainer,
        #     valX,
        #     image_name='Bee Swarm Plot.png'
        # )

        # plot_waterfall_plot()

        # plot of actual TOTALDEMAND values vs forecast values
        dates = valX.index
        plot_actual_vs_predicted(
            dates,
            valy,
            val_predictions,
            'validation_actual_vs_predicted.png'
        )


if __name__ == "__main__":
    main()
