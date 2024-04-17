"""

"""
from sklearn.metrics import mean_absolute_error
import xgboost as xgb
from .config import CFG


def evaluate_predictions(true_values, predictions):
    """
    Evaluate the accuracy of the predictions using MAE.
    """
    mae = mean_absolute_error(true_values, predictions)
    return mae
