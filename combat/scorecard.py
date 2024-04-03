# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 15:36:39 2024

@author: bidzh
"""

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt


        
def ScoreCard(y_proba: np.ndarray
              , log: bool
              , target_score: int = 600
              , target_odds: int = 30
              , pdo: int = 20
              ) -> pd.DataFrame:
    
    """
    The function calculates the scorecard based on the log odds
    
    Parameters:
    -----------
        
        y_proba: np.ndarray
            a numpy ndarray with probabilities of default. If log == True then ln(probabilities)
            
        log: bool
            whether y_proba is ln(probabilities) or not
    
        target_score: int, default = 600
            a baseline score
            
        target_odds: int, default = 30
            the corresponding odds for the target_score
            
        pdo: int, default = 20
            points to double odds
            
    Returns:
    --------
        pred: pd.DataFrame
            a pandas DataFrame with probability and the corresponding Score
    """
    # =============================================================================
    # Validating parameters    
    # =============================================================================
   
    if not isinstance(y_proba, np.ndarray):
        raise ValueError("""The 'y_prob' parameter must be np.ndarray""")

    if not isinstance(log, bool):
        raise TypeError("""The 'log' parameter must be logical""")
    
    if not isinstance(target_score, int) or target_score <= 0:
        raise ValueError("""The 'target_score' must be positive integer; got {}""".format(target_score))

    if not isinstance(target_odds, int) or target_odds <= 0:
        raise ValueError("""The 'target_odds' must be positive integer; got {}""".format(target_odds))

    if not isinstance(pdo, int) or pdo <= 0:
        raise ValueError("""The 'pdo' must be positive integer; got {}""".format(pdo))

    # =============================================================================
    # Calculating Score    
    # =============================================================================
    
    if not log:
        y_logproba = np.log(y_proba)
    else:
        y_logproba = y_proba
        y_proba = np.exp(y_proba)
    
    factor = pdo/np.log(2)
    offset = target_score - factor * np.log(target_odds)
    
    scorecard = pd.Series(offset - factor * y_logproba).round() 
    # scorecard.index = y_proba.index
    
    pred = pd.DataFrame({
            'Probability': y_proba
            , "Score": scorecard
        })
    
    return pred

