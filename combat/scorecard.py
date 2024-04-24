# -*- coding: utf-8 -*-
"""
The Scorecard module in this COMBAT package offers a specialized function, ScoreCard, tailored for constructing and managing scorecards, a common tool used in credit risk assessment. 
By providing a streamlined approach to scorecard development, this module empowers users to create transparent and interpretable models for credit scoring and risk assessment applications.

Key Fucntion of the Scorecard Module:

1. `ScoreCard(y_proba, log, target_score, target_odds, pdo)` -  serves as the centerpiece of the module, offering a user-friendly interface for building scorecards from predictive models

The Scorecard module provides a valuable resource for organizations seeking to implement transparent and interpretable models for credit scoring, risk assessment, and related applications
"""

import pandas as pd
import numpy as np
from typing import Union
        
def ScoreCard(y_proba: Union[np.ndarray, pd.Series]
              , log: bool
              , target_score: int = 600
              , target_odds: int = 30
              , pdo: int = 20
              ) -> pd.DataFrame:
    
    """
    The function calculates the scorecard based on the log odds
    
    Parameters:
    -----------
        
        y_proba: np.ndarray or pd.Series
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
   
    if not isinstance(y_proba, (np.ndarray, pd.Series)):
        raise TypeError("""The 'y_prob' parameter must be np.ndarray or pd.Series""")

    if isinstance(y_proba, np.ndarray) and len(y_proba.shape) != 1:
        raise ValueError("""The 'probabilities' parameter must have 1 column; got {}""".format(y_proba.shape))

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
    
    y_proba = np.array(y_proba)
    
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

