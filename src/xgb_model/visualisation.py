"""

"""
import matplotlib.pyplot as plt
import shap
from .config import CFG
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


def prepare_shap_explanations(model, X):
    """
    Prepare SHAP values and explanation objects.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Prepare the SHAP Explanation object for the full dataset
    explanation = shap.Explanation(
        values=shap_values,
        base_values=explainer.expected_value,
        data=X,
        feature_names=X.columns.tolist()
    )
    return explainer, explanation


def plot_shap_summary(explainer, X, image_name, instance_index=0):
    """
    Generate and save SHAP summary plots, excluding the feature with the highest absolute SHAP value.
    Handles both DataFrame and numpy array inputs for X with appropriate feature name handling.
    """
    # Calculate SHAP values for all features
    shap_values = explainer.shap_values(X)

    if isinstance(X, pd.DataFrame):
        feature_names = X.columns.tolist()
    else:  # Assuming X is a numpy array if not a DataFrame
        feature_names = [f"Feature {i}" for i in range(X.shape[1])]

    # Determine the index of the feature with the highest absolute SHAP value for the specified instance
    max_shap_value_index = np.argmax(np.abs(shap_values[instance_index]))

    # Delete the SHAP values for the highest impact feature
    if isinstance(shap_values, list):  # Handle multi-output models
        shap_values = [np.delete(sv, max_shap_value_index, axis=1) for sv in
                       shap_values]
    else:
        shap_values = np.delete(shap_values, max_shap_value_index, axis=1)

    # Drop the feature from the data and update feature names
    if isinstance(X, pd.DataFrame):
        X_modified = X.drop(columns=[feature_names[max_shap_value_index]])
        feature_names_modified = [name for i, name in enumerate(feature_names)
                                  if i != max_shap_value_index]
    else:
        X_modified = np.delete(X, max_shap_value_index, axis=1)
        feature_names_modified = [name for i, name in enumerate(feature_names)
                                  if i != max_shap_value_index]

    # Prepare the figure with the specified figure size
    plt.figure(figsize=(30, 10))

    # Generate SHAP summary plot without the highest impact feature
    shap.summary_plot(shap_values, X_modified,
                      feature_names=feature_names_modified, plot_type="bar",
                      show=False)

    # Save the plot to the specified file
    plt.savefig(os.path.join(CFG.images_path, image_name), bbox_inches='tight')
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
    shap_values_instance = shap_values[instance_index, :]
    feature_data_instance = X.iloc[instance_index, :]

    shap.decision_plot(explainer.expected_value, shap_values_instance, feature_data_instance, feature_names=X.columns.tolist())
    plt.savefig(os.path.join(CFG.images_path, image_name))
    plt.close()


def plot_beeswarm_plot(explanation, image_name):
    """
    Generate and save SHAP beeswarm plots using an SHAP Explanation object.
    """
    shap.plots.beeswarm(explanation)
    fig = plt.gcf()
    # fig.set_size_inches(CFG.img_dim1, CFG.img_dim2)
    # plt.savefig(os.path.join(CFG.images_path, image_name))
    plt.close()


def plot_waterfall_plot(explanation, image_name, figsize=(12, 8)):
    """
    Generate and save SHAP waterfall plots using a prepared SHAP Explanation object.

    Parameters:
    explanation (shap.Explanation): SHAP Explanation object for the instance to be plotted.
    image_name (str): The filename where the plot should be saved.
    figsize (tuple): Figure size for the plot.
    """
    # generate the waterfall plot
    shap.plots.waterfall(explanation)

    # adjust the layout and save the plot
    plt.tight_layout()
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
    # Calculate SHAP values directly using the passed explainer
    shap_values = explainer.shap_values(X)

    # Optionally exclude a feature
    if feature_to_exclude:
        feature_index = X.columns.get_loc(feature_to_exclude)
        shap_values = np.delete(shap_values, feature_index, axis=1)
        X = X.drop(columns=[feature_to_exclude])

    # Plot SHAP values
    shap.summary_plot(shap_values, X, plot_type="bar")

    # Save the plot
    plt.savefig(os.path.join(CFG.images_path, image_name))
    plt.close()
