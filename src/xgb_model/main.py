"""

"""
from .data_processing import (load_train, load_test, load_val)
from .model import (
    initialize_wandb, train_and_evaluate_model, save_model,
    load_model, evaluate_model
)
from .visualisation import (
    plot_shap_summary, plot_dependence_plots, plot_decision_plot,
    plot_waterfall_plot, plot_beeswarm_plot, plot_actual_vs_predicted,
    plot_shap_values, prepare_shap_explanations,
    plot_feature_importance
)
from .config import CFG
import os
import shap


# todo: add wandb logging, pytest, exception handling
def main():
    """
    main function to train and evaluate the model
    """
    if CFG.train:

        # load the training and test data
        dtrain, trainy = load_train()
        dtest, testy = load_test()

        # initialize W&B
        run = initialize_wandb()

        # train and evaluate the model
        model, mae = train_and_evaluate_model(dtrain, dtest, CFG.params)
        print(f"Mean Absolute Error on Test Set: {mae}")

        # save the model
        save_model(model)

        # finish W&B run
        if run:
            run.finish()
        return model, mae

    else:
        # load the validation data
        dval, valy, valX = load_val()

        # load the model
        model = load_model(os.path.join(CFG.models_path, 'xgb_model.json'))

        # evaluate the model
        val_mae, val_predictions = evaluate_model(model, dval, valy)
        print(f"Validation MAE: {val_mae}")

        return model, dval, valy, valX, val_predictions


if __name__ == "__main__":
    results = main()

    if not CFG.train:
        model, dval, valy, valX, val_predictions = results
        explainer, explanation = prepare_shap_explanations(model, valX)
        explainer = shap.Explainer(model)
        shap_values = explainer.shap_values(valX)
        dates = valX.index

        plot_actual_vs_predicted(
            dates,
            valy,
            val_predictions,
            'validation_actual_vs_predicted.png'
        )

        plot_feature_importance(
            model,
            'Feature Importance.png'
        )

        plot_shap_summary(
            model,
            valX,
            'SHAP summary.png'
        )

        plot_dependence_plots(
            explainer,
            valX,
            0,
            'dependence_plot_feature_0.png'
        )

        plot_dependence_plots(
            explainer,
            valX,
            1,
            'dependence_plot_feature_1.png'
        )

        plot_decision_plot(
            explainer,
            valX,
            explanation.values,
            'decision plot.png'
        )

        plot_waterfall_plot(
            explanation[1],
            'Waterfall Plot.png'
        )

        plot_shap_values(
            valX,
            explainer.shap_values(valX),
            shap_values,
            'SHAP Values.png'
        )

        plot_beeswarm_plot(
            explanation,
            'Bee Swarm Plot.png'
        )
