"""

"""
import matplotlib.pyplot as plt
import shap
from config import CFG
import os
import xgboost as xgb
import numpy as np


def plot_shap_summary(explainer, X, excluded=False):
    """
    Generate and save SHAP summary plots.
    """
    shap_values = explainer.shap_values(X)
    shap.summary_plot(shap_values, X, plot_type="bar")
    suffix = "_excluded" if excluded else ""
    plt.savefig(f"{CFG.images_path}/shap_summary_plot{suffix}.png")
    plt.close()


def plot_dependence_plots(explainer, X, feature_index):
    """
    Generate and save SHAP dependence plots for specific features.
    """
    shap_values = explainer.shap_values(X)
    shap.dependence_plot(feature_index, shap_values, X)
    plt.savefig(f"{CFG.images_path}/shap_dependence_plot_{feature_index}.png")
    plt.close()


def plot_decision_plot(explainer, X, instance_index=0, excluded=False):
    """
    Generate and save SHAP decision plots.
    """
    shap_values = explainer.shap_values(X)
    shap.decision_plot(explainer.expected_value, shap_values[instance_index, :], X.iloc[instance_index, :])
    suffix = "_excluded" if excluded else ""
    plt.savefig(f"{CFG.images_path}/shap_decision_plot{suffix}.png")
    plt.close()


def plot_waterfall_plot(explainer, X, instance_index=0, excluded=False):
    """
    Generate and save SHAP waterfall plots.
    """
    shap_values = explainer.shap_values(X)
    expl = shap.Explanation(
        values=shap_values[instance_index],
        base_values=explainer.expected_value,
        data=X.iloc[instance_index],
        feature_names=X.columns
    )
    shap.waterfall_plot(expl)
    suffix = "_excluded" if excluded else ""
    plt.savefig(f"{CFG.images_path}/shap_waterfall_plot{suffix}.png")
    plt.close()


def plot_beeswarm_plot(explainer, X, excluded=False):
    """
    Generate and save SHAP beeswarm plots.
    """
    shap_values = explainer.shap_values(X)
    shap.plots.beeswarm(shap_values)
    suffix = "_excluded" if excluded else ""
    plt.savefig(f"{CFG.images_path}/shap_beeswarm_plot{suffix}.png")
    plt.close()


def plot_actual_vs_predicted(dates, actual, predicted):
    """
    Plot and save the comparison of actual vs predicted values.
    """
    plt.figure(figsize=(CFG.img_dim1, CFG.img_dim2))
    plt.plot(dates, actual, label='Actual', marker='.', linestyle='-')
    plt.plot(dates, predicted, label='Predicted', marker='.', linestyle='--')
    plt.title('Validation Set: Actual vs Predicted Demand')
    plt.xlabel('Date Time')
    plt.ylabel('Total Demand')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{CFG.images_path}/actual_vs_predicted.png")
    plt.close()


def plot_feature_importance(model):
    """
    Plot and save feature importance.
    """
    xgb.plot_importance(model)
    plt.savefig(CFG.images_path + '/feature_importance.png')
    plt.close()


def initialize_shap_explainer(model, X):
    """
    Initialize a SHAP TreeExplainer and calculate SHAP values.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    return explainer, shap_values


def create_explanation_objects(explainer, X, shap_values, feature_to_exclude=None):
    """
    Create SHAP Explanation objects for individual and multiple observations.
    """
    if feature_to_exclude:
        feature_index = X.columns.get_loc(feature_to_exclude)
        shap_values_excluded = np.delete(shap_values, feature_index, axis=1)
        X_excluded = X.drop(columns=[feature_to_exclude])
        shap_values_instance_excluded = shap_values_excluded[0, :]
        X_instance_excluded = X_excluded.iloc[0, :]

        expl_instance_excluded = shap.Explanation(
            values=shap_values_instance_excluded,
            base_values=explainer.expected_value,
            data=X_instance_excluded,
            feature_names=X_excluded.columns.tolist()
        )
        return expl_instance_excluded, X_excluded, shap_values_excluded
    else:
        expl_instance = shap.Explanation(
            values=shap_values[0],
            base_values=explainer.expected_value,
            data=X.iloc[0],
            feature_names=X.columns
        )
        expl_multi = shap.Explanation(
            values=shap_values,
            base_values=explainer.expected_value,
            data=X,
            feature_names=X.columns
        )
        return expl_instance, expl_multi


def plot_shap_values(explainer, X, feature_to_exclude=None):
    """
    Plot various SHAP visualizations.
    """
    expl_instance, expl_multi = create_explanation_objects(explainer, X, explainer.shap_values(X), feature_to_exclude)
    shap.summary_plot(expl_instance.values, expl_instance.data, plot_type="bar")
    shap.plots.beeswarm(expl_multi)