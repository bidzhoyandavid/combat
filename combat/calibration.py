# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 14:48:56 2024

@author: bidzh
"""

from combat.models import LogitModel
from combat.short_list import *
from combat.combat import *

import pandas as pd
import numpy as np
from tqdm import tqdm

from itertools import combinations

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score
                            , precision_recall_curve
                            , f1_score
                            , auc
                            , roc_curve
                            , accuracy_score
                            , brier_score_loss
                            , RocCurveDisplay
                            , confusion_matrix
                            )
from sklearn.calibration import calibration_curve

import matplotlib.pyplot as plt


def ExpectedCalibrationError(true_labels: pd.Series
                             , probabilities: np.ndarray
                             , n_bins: int = 20
                             ) -> float:
    
    """
    The function calculates the Expected Calibration Error
    
    Parameters:
    ----------
        true_labels: pd.Series
            true labels
            
        probabilities: np.ndarray
            probabilities of each object in the dataset
            
        n_bins: int
            number of bins
            
    Returns:
    --------
        ece: float
            expected calibration error
    """
    # =============================================================================
    # Validating parameters
    # =============================================================================
    if not isinstance(true_labels, pd.Series):
        raise TypeError("The 'labels' parameter must be pd.Series")
        
    if not isinstance(probabilities, np.ndarray):
        raise TypeError("The 'probabilities' parameter must be pd.Series")
        
    if not isinstance(n_bins, int):
        raise TypeError("The 'n_bins' parameter must be an integer")
        
    if n_bins <= 0:
        raise ValueError("""The 'n_bins' parameter must be positive integer""")

    # =============================================================================
    # Calculating ECE
    # =============================================================================
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    confidences = np.max(probabilities, axis=1)
    predicted_label = np.argmax(probabilities, axis=1)

    accuracies = predicted_label == true_labels.values

    ece = np.zeros(1)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = np.logical_and(confidences > bin_lower.item(), confidences <= bin_upper.item())
        prob_in_bin = in_bin.mean()

        if prob_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prob_in_bin
    return round(float(ece), 4)

   
def CalibrationModel(
        x_data: pd.DataFrame
        , y_data: pd.Series
        , penalty: float = None
        , alpha: float = 0.5
        , fit_intercept: bool = True
        ):
   
    """
    The function creates the calibrated model
    
    Parameters:
    ----------
        x_data: pd.DataFrame
            a pandas DataFrame with explanatory variables
            
        y_data: pd.Series
            a pandas Series with discrete dependent variable
            
        penalty: str {'None', 'l1', 'l2'}, default = None
            a regularization for Logistic Regression model
            
        alpha: float, default = 0.5
            a float variable for regularization
            
        fit_intercept: bool, {'True', 'False'}, default = 'True'
            a bool variable whether to fit intercept in the Logistic Regression or not         

    Returns:
    --------
        model: Logistic Regression Model        
    """
    
    # =============================================================================
    # Validating parameters    
    # =============================================================================
    if not isinstance(x_data, pd.DataFrame):
        raise TypeError("""The 'x_data' parameter must be a pandas DataFrame object""")
    
    if not isinstance(y_data, pd.Series):
        raise TypeError("""The 'y_data' parameter must be a pandas Series""")
    
    if len(x_data) != len(y_data):
        raise ValueError("""The length of 'x_data' and 'y_data' must be identical""")

    if penalty not in [None, 'l1', 'l2']:
        raise ValueError("""The 'penalty' parameter must be in ['None', 'l1', 'l2']""")
    
    if not isinstance(alpha, float):
        raise TypeError("""The 'alpha' parameter must be float""")
    
    if not 0 < alpha < 1:
        raise ValueError("""The 'alpha' parameter must be between 0 and 1""")
        
    if not isinstance(fit_intercept, bool):
        raise TypeError("""The 'fit_intercept' must be logical""")

    # =============================================================================
    # Creating Calibration Model    
    # =============================================================================
    model = LogisticRegression(
        penalty = penalty
        , C = alpha
        , fit_intercept = fit_intercept
        , solver='saga'
        ).fit(x_data, y_data)
    
    return model

def PredictionCalibration(
        x_data: pd.DataFrame
        , model: LogisticRegression 
        , logprob: bool = False
        ) -> pd.Series:
    
    """
    The function predicts the Probability based on the calibration model
    
    Parameters:
    ----------
        x_data: pd.DataFrame
            a series of predictions of the raw models
            
        model: CalibrationModel
            a Logistic Regression Model for calibration
            
        logrob: bool
            a boolean variable whether to calculate a logarithm of probabilities
            
    Returns:
    --------
        pred: pd.Series
            a pandas Series of the calibrated probabilities
    """
    
    # =============================================================================
    # Validating parameters    
    # =============================================================================
    if not isinstance(x_data, pd.DataFrame):
        raise TypeError("""The 'x_data' parameter must be a pandas DataFrame object""")

    if not isinstance(logprob, bool):
        raise TypeError("""The 'logprob' parameter must be logical""")
        
    # =============================================================================
    # Calculating prediction    
    # =============================================================================
    if logprob:
        pred = model.predict_log_proba(x_data)[:,1]
    else:
        pred = model.predict_proba(x_data)[:,1]
        
    pred = pd.Series(pred).round(4)
    
    return pred
    
def CalibrationCurve(
        y_data: pd.Series
        , probabilities: pd.Series
        , n_bins: int
        , label: str
        ):
    """
    The function plots the calibration curve
    
    Parameters:
    -----------
        y_data: pd.Series
            a pandas Series with discrete dependent variable
         
        probabilities: np.ndarray
            a numpy ndarray with estimated probabilities of default
            
        n_bins: int
            a numbeer ob bins
            
        label: str
            a label of the plot
    """
    # =============================================================================
    # Validating parameters    
    # =============================================================================
    if not isinstance(y_data, pd.Series):
        raise TypeError("""The 'y_data' parameter must be a pandas Series""")
    
    if not isinstance(probabilities, np.ndarray):
        raise TypeError("""The 'probabilities' parameter must be a pandas Series""")
        
    if not isinstance(n_bins, int):
        raise TypeError("""The 'n_bins' parameter must be an integer""")
    
    if n_bins <= 0:
        raise ValueError("""The 'n_bins' parameter must be positive integer""")
    
    if not isinstance(label, str):
        raise ValueError("""The 'label' parameter must be string""")

    # =============================================================================
    # Plot Calibration Curve   
    # =============================================================================

    x, y = calibration_curve(y_data, probabilities, n_bins=10)

    # Plot the calibration curve
    plt.plot([0,  1], [0,  1], linestyle='--', label='Ideally Calibrated')
    plt.plot(y, x, marker='.', label=label)

    # Add legend and labels
    plt.legend(loc='upper left')
    plt.xlabel('Average Predicted Probability in each bin')
    plt.ylabel('Ratio of positives')
    plt.show(block=False)
    


















