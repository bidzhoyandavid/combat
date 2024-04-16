# -*- coding: utf-8 -*-
"""
The Utilities module in this COMBAT package provides a collection of functions designed to streamline various data preprocessing and model evaluation tasks. 
With functionalities like variable deletion, model selection, and prediction result reporting, the Utilities module serves as a versatile toolkit for enhancing efficiency and productivity in data science workflows.


Functions within the Utilities module:

1. `DeleteVars(x_train, x_test, df_sign, vars_to_remove)` - facilitate the removal of specified variables from input datasets and dataframes    
2. `SelectModels(models_dict, meta_data, select_by, cutoff)` -  aids in the selection of models from a dictionary with LogitModel objects.
3. `PredictionResults(y_data, probabilities, cutoff)` - function offers a convenient solution for reporting prediction results
"""


import pandas as pd
import numpy as np
from typing import Union

from sklearn.metrics import (roc_auc_score
                            # , precision_score
                            # , recall_score
                            # , f1_score
                            # , auc
                            # , roc_curve
                            # , accuracy_score
                            # , brier_score_loss
                            # , confusion_matrix
                            , classification_report
                            )


def DeleteVars(
        x_train: pd.DataFrame
        , x_test: pd.DataFrame
        , df_sign: pd.DataFrame
        , vars_to_remove: list      
        ) -> dict:
    
    """
    The function removes variables to be deleted from training and testing sets and 
    removes from the dataframe of sign expectation
    
    Parameters:
    -----------
        x_train: pd.DataFrame
            a pandas DataFrame with training data set
            
        x_test: pd.DataFrame
            a pandas DataFrame with testing data set
            
        df_sign:
            a pandas Dataframe with coefficients expectation
            
        vars_to_remove: list
            a list of variables to be removed
            
    Returns:
    --------
        final_data: dict
            keys:
                x_train_new
                x_test_new
                df_sign_new
    """
    
    # =============================================================================
    # validating parameters    
    # =============================================================================
    
    if set(x_test.columns) != set(x_train.columns):
        raise ValueError("""The columns of 'x_test' and 'x_train' must be identical""")
    
    if not isinstance(x_train, pd.DataFrame):
        raise ValueError("""The 'x_train' parameter must be a pandas DataFrame""")
    
    if len(x_train.columns) != len(x_test.columns):
        raise ValueError("""The number of columns in 'x_train' and 'x_test' must be identical""")
    
    if not isinstance(x_test, pd.DataFrame):
        raise ValueError("""The 'x_test' parameter must be a pandas DataFrame""")
    
    if not isinstance(df_sign, pd.DataFrame):
        raise ValueError("""The 'df_sign' parameter must be a pandas DataFrame""")

    if not isinstance(vars_to_remove, list):
        raise ValueError("""The 'vars_to_remove' must be a list""")
    
    if len(x_train.columns) != len(df_sign):
        raise ValueError("""The number of 'x_train' columns must be identical with legnth of 'df_sign'""")
        
    if not all(i in x_train.columns for i in vars_to_remove ):
        raise ValueError("""The items of 'vars_to_remove' parameter must be in x_train.columns""")
        
    # =============================================================================
    # Removing variables    
    # =============================================================================
    
    x_train_new = x_train.drop(columns = vars_to_remove)
    x_test_new = x_test.drop(columns = vars_to_remove)
    
    df_sign_new = df_sign.drop(vars_to_remove)
    
    final_data = {
        "x_train_new": x_train_new
        , "x_test_new": x_test_new
        , "df_sign_new": df_sign_new
        }
    
    return final_data
    
    
    
def SelectModels(
        models_dict: dict
        , meta_data: pd.DataFrame
        , select_by: str = 'gini_test'   
        , cutoff: float = 0.5
        )  -> dict:
    
    """
    The function filters models from dictionary of LogitModel instances
    
    Paramteres:
    -----------
        models_dict: dict
            a dictionary with LogitModel instances
            
        meta_data:
                a pandas DataFrame with meta information of all models in the dictionary.
                
        select_by: str {'gini_test', 'gini_train'
                        , 'auc_test', 'auc_train'
                        , 'Brier_test', 'Brier_train'
                        , 'F1_test', 'F1_train'}, default = 'gini_test'
            a metric to filter by models
        
        cutoff: float
            a cutoff value for the 'select_by' metric
            
    Returns:
    --------
        new_models_dict: dict
            a dictionary with selected models
    """
    
    # =============================================================================
    # Validating parameters    
    # =============================================================================
    
    if not isinstance(models_dict, dict):
        raise TypeError("""The 'model_dict' parameter must a dictionary""")
    
    if select_by not in ['gini_test', 'gini_train'
                       , 'auc_test', 'auc_train'
                       , 'Brier_test', 'Brier_train'
                       , 'F1_test', 'F1_train']:
        raise ValueError("""The 'select_by' parameter must be in  ['gini_test', 'gini_train', 'auc_test', 'auc_train'
                                                                     , 'Brier_test', 'Brier_train', 'F1_test', 'F1_train']; got {}""".format(select_by))
      
    if not isinstance(meta_data, pd.DataFrame):
        raise TypeError("""The 'meta_data' must be a pandas DataFrame""")
        
    if len(meta_data) != len(models_dict):
        raise ValueError("""The length of 'meta_data' and 'models_dict' must be identical""")
       
    if not isinstance(cutoff, float) or not 0 < cutoff < 1:
        raise ValueError("""The 'cutoff' must be positive float from 0 to 1; got {}""".format(cutoff))
       
    # =============================================================================
    # Selecting models        
    # =============================================================================
    
    if select_by in ['Brier_test', 'Brier_train']:
        meta_new = meta_data[meta_data[select_by] < cutoff]        
    else:
        meta_new = meta_data[meta_data[select_by] > cutoff]
        
    new_models_dict = {key: models_dict[key] for key in meta_new.index}
    
    return new_models_dict
        
    
    
def PredictionResults(
        y_data: pd.Series
        , probabilities: Union[pd.Series, np.ndarray]
        , cutoff: float
        ) -> dict:
    """
    The function prints the key metrics of the predicted scores against true labels
    
    Parameters:
    -----------
        y_data: pd.Series
            a pandas Series with true labels
            
        probabilities: pd.Series or np.ndarray
            a pandas Series with predicted probabilities
            
        cutoff: float
            a cutoff value the classify object 
        
    """

    # =============================================================================
    # Validating parameters    
    # =============================================================================

    if not isinstance(y_data, pd.Series):
        raise TypeError("""The 'y_data' parameter must be a pandas Series""")
        
    if not isinstance(probabilities, (pd.Series, np.ndarray)):
        raise TypeError("""The 'probabilities' parameter must be a pandas Series""")
      
    if isinstance(probabilities, np.ndarray) and len(probabilities.shape) != 1:
        raise ValueError("""The 'probabilities' parameter must have 1 column; got {}""".format(probabilities.shape))
      
    if len(y_data) != len(probabilities):
        raise ValueError("""The length of 'y_data' and 'probabilities' parameters must be identitcal""")
        
    if not isinstance(cutoff, float) or not 0 < cutoff < 1:
        raise ValueError("""The 'cutoff' must be positive float from 0 to 1; got {}""".format(cutoff))
        
    # =============================================================================
    # Printing results    
    # =============================================================================

    probabilities = pd.Series(probabilities)    

    auc_score = roc_auc_score(y_data, probabilities)
    auc_score =  float("{:.3f}".format(auc_score))
    
    gini = 2 * auc_score -1
    gini = float("{:.3f}".format(gini))
    
    labels = [1 if prob > cutoff else 0 for prob in probabilities]
    
    
    print("""Gini: {}""".format(gini), "||| AUC: {}".format(auc_score))
    # print("""AUC: {}""".format(auc_score))
    target_names = ['Defaulted', 'Non-defaulted']
    print(classification_report(y_true = y_data
                                , y_pred = labels
                                , target_names=target_names)
          )
    
    

 