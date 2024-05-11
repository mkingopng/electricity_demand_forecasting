"""

"""
import matplotlib.pyplot as plt
import shap
from .config import CFG
import os
import xgboost as xgb
import numpy as np


def initialize_shap_explainer(model, X):
    """
    Initialize a SHAP TreeExplainer and calculate SHAP values.
    """
    explainer = shap.Explainer(model)
    shap_values = explainer.shap_values(X)
    return explainer, shap_values


def prepare_shap_explanations(model, X):
    """
    Prepare SHAP values and explanation objects.
    Assumes X is a Pandas DataFrame.
    :param model: The trained model.

    """
    # initialize the SHAP explainer
    explainer = shap.TreeExplainer(model, feature_perturbation='tree_path_dependent')

    # compute SHAP values
    shap_values = explainer.shap_values(X)

    # prepare the SHAP Explanation object for the full dataset
    explanation = shap.Explanation(
        values=shap_values,
        base_values=explainer.expected_value,
        data=X,
        feature_names=X.columns.tolist()
    )
    return explainer, explanation


def plot_shap_summary(model, X, image_name):
    """
    Generate and save SHAP summary plots. Assumes X is a DataFrame.
    :param model: The trained model.
    :param X: The input features.
    :param image_name: The filename for the saved plot.
    :return: None
    """
    # initialize the SHAP explainer and calculate SHAP values directly
    explainer = shap.Explainer(model)
    shap_values = explainer(X)

    # generate and save the SHAP summary plot
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    plt.savefig(os.path.join(CFG.images_path, image_name), bbox_inches='tight')
    plt.show()
    plt.close()


def plot_dependence_plots(explainer, X, feature_index, image_name):
    """
    generate and save SHAP dependence plots for specific features
    :param explainer: (shap.Explainer) SHAP explainer object
    :param X: (pd.DataFrame) DataFrame of input features
    :param feature_index: (int) Index of the feature to plot
    :param image_name: (str) The filename where the plot should be saved
    """
    shap_values = explainer.shap_values(X)
    shap.dependence_plot(feature_index, shap_values, X)
    plt.savefig(os.path.join(CFG.images_path, image_name))
    plt.show()
    plt.close()


def plot_decision_plot(explainer, X, shap_values, image_name, instance_index=0):
    """
    generate and save SHAP decision plots for a specific instance
    :param explainer: (shap.Explainer) SHAP explainer object
    :param X: (pd.DataFrame) DataFrame of input features
    :param shap_values: (np.array) SHAP values for the input features
    :param image_name: (str) The filename where the plot should be saved
    """
    shap_values_instance = shap_values[instance_index, :]  # access the specific instance's SHAP values
    feature_data_instance = X.iloc[instance_index, :]  # access the specific instance's feature data
    shap.decision_plot(
        explainer.expected_value,  # use expected_value from the explainer
        shap_values_instance,  # use the specific instance's SHAP values
        feature_data_instance,  # use the specific instance's feature data
        feature_names=X.columns.tolist()  # ensure feature names are passed correctly
    )
    plt.savefig(os.path.join(CFG.images_path, image_name))
    plt.show()
    plt.close()


def plot_beeswarm_plot(explanation, image_name):
    """
    generate and save SHAP beeswarm plots using an SHAP Explanation object
    :param explanation: (shap.Explanation) SHAP Explanation object
    :param image_name: (str) The filename where the plot should be saved
    """
    shap.plots.beeswarm(explanation)
    fig = plt.gcf()
    # fig.set_size_inches(CFG.img_dim1, CFG.img_dim2)
    plt.savefig(os.path.join(CFG.images_path, image_name))
    plt.show()
    plt.close()


def plot_waterfall_plot(explanation, image_name, figsize=(12, 8)):
    """
    generate and save SHAP waterfall plots using a prepared SHAP Explanation
    object
    :param explanation: (shap.Explanation) SHAP Explanation object for the
    instance to be plotted
    :param image_name: (str) The filename where the plot should be saved.
    :param figsize: (tuple) Figure size for the plot.
    """
    shap.plots.waterfall(explanation)
    plt.tight_layout()
    plt.savefig(os.path.join(CFG.images_path, image_name))
    plt.show()
    plt.close()


def plot_actual_vs_predicted(dates, actual, predicted, image_name):
    """
    Plot and save the comparison of actual vs predicted values.
    :param dates: The dates for the actual and predicted values.
    :param actual: The actual values.
    :param predicted: The predicted values.
    :param image_name: The filename for the saved plot.
    :return: None
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


def plot_feature_importance(model, image_name, max_features=30):
    """
    plot and save feature importance, optionally showing only the top N features.
    :param model: The trained model.
    :param image_name: The filename for the saved plot.
    :param max_features: The maximum number of top features to display.
    :returns: None
    """
    ax = xgb.plot_importance(model, max_num_features=max_features, height=0.5, importance_type='weight')
    ax.figure.set_size_inches(10, 8)  # Adjust plot size
    plt.title('Feature Importance')
    plt.xlabel('F score', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.yticks(fontsize=10)
    plt.xticks(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(CFG.images_path, image_name))
    plt.show()
    plt.close()


def plot_shap_values(X, shap_values, image_name, feature_to_exclude=None):
    """
    Plot SHAP values for the given feature data
    :param X: (pd.DataFrame) The feature data.
    :param shap_values: (np.array) The SHAP values.
    :param image_name: (str) The filename for the saved plot.
    :param feature_to_exclude: (str) Optional feature to exclude from the plot.
    :returns:
    """
    # optionally exclude a feature
    if feature_to_exclude and feature_to_exclude in X.columns:
        feature_index = X.columns.get_loc(feature_to_exclude)
        shap_values = np.delete(shap_values, feature_index, axis=1)
        X = X.drop(columns=[feature_to_exclude])

    # ensure shap_values is an array or matrix, not single value or explainer
    if isinstance(shap_values, (np.ndarray, list)) and len(shap_values) > 0:
        shap.summary_plot(shap_values, X, plot_type="bar")
        # plt.savefig(os.path.join(CFG.images_path, image_name))
        plt.show()
        plt.close()
    else:
        raise ValueError("SHAP values must be a non-empty array or matrix.")
