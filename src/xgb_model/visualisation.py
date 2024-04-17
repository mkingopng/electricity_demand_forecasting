"""

"""
import matplotlib.pyplot as plt
import shap
from config import CFG
import os
import xgboost as xgb
import numpy as np
import pandas as pd


def initialize_shap_explainer(model, X):
    """
    Initialize a SHAP TreeExplainer and calculate SHAP values.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    return explainer, shap_values


def create_explanation_objects(explainer, X, shap_values, feature_to_exclude=None):
    """
    create SHAP Explanation objects for individual and multiple observations
    """
    shap_values = explainer.shap_values(X)
    if feature_to_exclude and feature_to_exclude in X.columns:
        feature_index = X.columns.get_loc(feature_to_exclude)
        shap_values_excluded = np.delete(shap_values, feature_index, axis=1)
        X_excluded = X.drop(columns=[feature_to_exclude])
        return X_excluded, shap_values_excluded
    else:
        return shap_values, X  # Return original SHAP values and DataFrame


def create_explanation_object(explainer, X, feature_to_exclude=None):
    """
    Generate SHAP Explanation objects, possibly excluding specific features.
    """
    # make sure X is in the correct format
    if not isinstance(X, pd.DataFrame):
        raise ValueError("X must be a pandas DataFrame.")

    print("Shape of X before SHAP calculation:", X.shape)

    # this should not convert X if it's already a df
    shap_values = explainer.shap_values(X)

    if feature_to_exclude and feature_to_exclude in X.columns:
        feature_index = X.columns.get_loc(feature_to_exclude)
        shap_values_excluded = np.delete(shap_values, feature_index, axis=1)
        X_excluded = X.drop(columns=[feature_to_exclude])
        explanation = shap.Explanation(
            values=shap_values_excluded,
            base_values=explainer.expected_value,
            data=X_excluded,
            feature_names=X_excluded.columns.tolist()
        )
    else:
        explanation = shap.Explanation(
            values=shap_values,
            base_values=explainer.expected_value,
            data=X,
            feature_names=X.columns.tolist()
        )

    return explanation


def plot_shap_summary(explainer, X, image_name, excluded=False):
    """
    Generate and save SHAP summary plots.
    """
    shap_values = explainer.shap_values(X)
    shap.summary_plot(shap_values, X, plot_type="bar")
    plt.savefig(os.path.join(CFG.images_path, image_name))
    plt.close()


def plot_dependence_plots(explainer, X, feature_index, image_name):
    """
    Generate and save SHAP dependence plots for specific features.
    """
    shap_values = explainer.shap_values(X)
    shap.dependence_plot(feature_index, shap_values, X)
    plt.savefig(os.path.join(CFG.images_path, image_name))
    plt.close()


def plot_decision_plot(explainer, X, shap_values, image_name, instance_index=0):
    """
    Generate and save SHAP decision plots for a specific instance.
    """
    # SHAP values for the specific instance
    shap_values_instance = shap_values[instance_index, :]
    # feature data for the specific instance
    feature_data_instance = X.iloc[instance_index, :]
    # create the decision plot using SHAP values for specific instance
    shap.decision_plot(explainer.expected_value, shap_values_instance, feature_data_instance)
    plt.savefig(os.path.join(CFG.images_path, image_name))
    plt.close()


def plot_waterfall_plot(explainer, X, image_name, instance_index=0):
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
    plt.savefig(os.path.join(CFG.images_path, image_name))
    plt.close()


def plot_beeswarm_plot(explainer, X, image_name):
    """
    Generate and save SHAP beeswarm plots.
    """
    # Ensure that X is a DataFrame and has the correct format
    if not isinstance(X, pd.DataFrame):
        raise ValueError("X must be a pandas DataFrame.")

    # Print the shape of X for debugging
    print("Shape of X at the start of plot_beeswarm_plot:", X.shape)

    shap_values = explainer.shap_values(X)
    explanation = create_explanation_object(explainer, X, shap_values)
    shap.plots.beeswarm(explanation)
    plt.savefig(os.path.join(CFG.images_path, image_name))
    plt.close()


def plot_actual_vs_predicted(dates, actual, predicted, image_name):
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
    plt.savefig(os.path.join(CFG.images_path, image_name))
    plt.show()
    plt.close()


def plot_feature_importance(model, image_name):
    """
    Plot and save feature importance.
    """
    xgb.plot_importance(model)
    plt.savefig(os.path.join(CFG.images_path, image_name))
    plt.close()


def plot_shap_values(explainer, X, image_name, feature_to_exclude=None):
    """
    Plot various SHAP visualizations, handling optional exclusion of a feature.
    """
    shap_values, X_adjusted = create_explanation_objects(explainer, X, feature_to_exclude)
    shap.summary_plot(shap_values, X_adjusted, plot_type="bar")
    plt.savefig(os.path.join(CFG.images_path, image_name))
    plt.close()
