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


def create_explanation_object(explainer, X, feature_to_exclude='FORECAST_DEMAND(t-1)'):
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


# Example of using the function
# plot_shap_summary(explainer, your_numpy_array, 'shap_summary.png')


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


def plot_waterfall_plot(explainer, X, image_name, instance_index=0, figsize=(12, 8)):
    """
    Generate and save SHAP waterfall plots, excluding the feature with the highest absolute SHAP value.
    """
    # Compute SHAP values
    shap_values = explainer.shap_values(X)
    feature_names = X.columns.tolist()

    # Determine the index of the feature with the highest absolute SHAP value for the specified instance
    max_shap_value_index = np.argmax(np.abs(shap_values[instance_index]))

    # Delete the SHAP values for the highest impact feature
    shap_values_modified = np.delete(shap_values, max_shap_value_index, axis=1)

    # Drop the feature from the data
    X_modified = X.drop(columns=[feature_names[max_shap_value_index]])

    # Remove the feature from the list of feature names
    feature_names_modified = [name for i, name in enumerate(feature_names) if i != max_shap_value_index]

    # Prepare the figure with the specified figure size
    plt.figure(figsize=(30, 10))

    # Create the SHAP explanation object without the highest impact feature
    expl = shap.Explanation(
        values=shap_values_modified[instance_index],
        base_values=explainer.expected_value,
        data=X_modified.iloc[instance_index],
        feature_names=feature_names_modified
    )

    # Generate and save the plot
    shap.waterfall_plot(expl)
    plt.subplots_adjust(left=1)
    plt.tight_layout()
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
