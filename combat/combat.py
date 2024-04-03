# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 22:38:37 2024

@author: bidzh
"""

from combat.models import LogitModel
from combat.short_list import *

import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import Union, Optional

import random 
from random import randrange
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



def IsModelValid(
            model: LogitModel
            , coef_expectation: pd.DataFrame
            , p_value: float = 0.05
            , check_sample: str = 'test'
            , metric: str = 'gini'
            , gini_cutoff: float = 0.5
            , auc_cutoff: float = 0.7
        ) -> bool:
    
    """
    The function calculates wether the model meets prespecified requirements
    
    Parameters:
    -----------
        model: LogitModel() 
            An object of LogitModel() class
                
        coef_expectation: pd.DataFrame() 
            a pandas DataFrame() object containing variable names and their sign expectations
                            
        p_value: float, default = 0.05
            variable significance level
                
        check_sample: str {'test', 'train'}, default = 'test'
            the sample to perform test
                        
        metric: str {'gini', 'auc'} default = 'gini' 
            a metric used to measure the model accuracy
                
        gini_cutoff: float, default = 0.4
            a cutoff value for gini
        
        auc_cutoff: float, default = 0.7
            a cutoff value for AUC
        
    Returns:
    --------
        final_result: bool
            Whether the model meets all the specifies requirements
    """
    
    # =============================================================================
    # Validating parameters    
    # =============================================================================
    if not isinstance(model, LogitModel):
        raise TypeError(""" The 'model' parameter must be LogitModel type""")
    
    if not isinstance(coef_expectation, pd.DataFrame):
        raise TypeError("""The 'coef_expectation' parameter must be a pandas DataFrame""")
           
    if check_sample not in ['train', 'test']:
        raise ValueError("""The 'check_sample' parameter must be in ['train', 'test']; got {}""". format(check_sample))
        
    if metric not in ['gini', 'auc']:
        raise ValueError("""The 'metric' parameter must be in ['gini', 'auc']; got {}""".format(metric))
        
    if not isinstance(p_value, float) or not 0 < p_value < 0.5:
        raise ValueError("""The 'p_value' parameter must be float from 0 to 0.5; got {}""".format(p_value))

    if not isinstance(gini_cutoff, float) or not 0 < gini_cutoff < 1:
        raise ValueError("""The 'gini_cutoff' parameter must be float from 0 to 1; got {}""".format(gini_cutoff))

    if not isinstance(auc_cutoff, float) or  not 0 < auc_cutoff < 1:
        raise ValueError("""The 'auc_cutoff' parameter must be from float 0 to 1; got {}""".format(auc_cutoff))
        
    # =============================================================================
    # Accuracy Check        
    # =============================================================================
    if metric != 'auc':
        method_name = metric.capitalize()  + '_' + check_sample.capitalize()
    else:
        method_name = metric.upper()  + '_' + check_sample.capitalize()
    
    accuracy_result = getattr(model, method_name)()
    if metric == 'gini':
        result = accuracy_result > gini_cutoff
    else:
        result = accuracy_result > auc_cutoff
        
    # economic constraint ---------------------------------
    coef_expectation.columns = ['variable', 'sign_expectation']
    coefs = model.GetCoefficients_SM()
    coefs = coefs.merge(coef_expectation
                        , how = 'inner'
                        , on = 'variable'
                        )
    coefs['check'] = coefs['coefficient'] * coefs['sign_expectation']
    coefs_check = coefs[coefs['check'] < 0]
    if len(coefs_check) == 0:
        coefs_check = True
    else:
        coefs_check = False
                
    # =============================================================================
    # p_value Check
    # =============================================================================
    coefs['p_value_check'] = coefs['p_value'].apply(lambda x: 1 if x > p_value else 0)
    coefs_pvalue = coefs[coefs['p_value_check'] == 1]
    
    if len(coefs_pvalue) == 0:
        coefs_pvalue = True
    else:
        coefs_pvalue = False
        
    # final result ---------------------------------------------
    if result and coefs_check and coefs_pvalue:
        return True
    else:
        return False
        


def ModelCombination(
            y_train: pd.Series
            , x_train: pd.DataFrame
            , y_test: pd.Series
            , x_test: pd.DataFrame
            , max_model_number: int 
            , dependent_number: int
            , coef_expectation: pd.DataFrame
            , intercept: bool = True
            , penalty: Optional[str] = None
            , alpha: float = 0.5
            , p_value: float = 0.05
            , check_sample: str = 'test'
            , metric: str = 'gini'
            , gini_cutoff: float = 0.5
            , auc_cutoff: float = 0.7                    
        ) -> dict:
    
    """
    The function creates many models and checks whether they accurate or not
    
    Parameters:
    -----------
        y_train: pd.Series()
            a series of binary dependent variable of train sample
            
        x_train: pd.DataFrame()
            a dataframe with explanatory variables of train sample
            
        y_test: pd.Series()
            a series of binary dependent variable of test sample
            
        x_test: pd.DataFrame()
            a dataframe with explanatory variables of test sample
            
        max_model_number: int
            maximum number of models to be created
        
        dependent_number: int
            quantity of dependent variable to input in the model. The number of coef_expectation
            must be less than the columns number of x_train
            
        coef_expectation: pd.DataFrame()
            a dataframe with variable names and their sign expectations. The name of variables must 
            be the same as x_train and x_test column names
         
        intercept: bool
            an indicator whether to include intercept into model or not 

        penalty: str {None, 'l1'}, default = None
            regularization parameter. Only 'l1' regularization is available   

        alpha: float
            regularization parameter from 0 to 1
            
        p_value: float, default = 0.05
            max significance level
            
        check_sample: str {'test', 'train'}, default = 'test'
            a sample to calculate key metrics
        
        metric: str {'gini', 'auc', 'accuracy'}, default = 'gini'
            metric to calculate 
        
        gini_cutoff: float, defualt = 0.5
            a cutoff value of Gini 
            
        auc_cutoff: float, default = 0.7
            a cutoff value of AUC
        
        accuracy_cutoff: float, default = 0.7
            a cutoff value of accuracy ratio
    
    
    Returns:
    -------
        final_data:
            key: the number of model
            value: model
    """

    # =============================================================================
    # Validating parameters    
    # =============================================================================
    if len(coef_expectation.columns) != 2:
        coef_expectation = coef_expectation.reset_index()
    coef_expectation.columns = ['variable', 'sign_expectation']

    if not set(x_train.columns) == set(coef_expectation['variable']): 
        raise ValueError("""The 'x_train' columns are not identical with 'coef_expectation'""")
      
    if len(x_train.columns) < dependent_number:
        raise ValueError("The 'dependent_number' must be less then number of x_train columns quantity")
    
    if not isinstance(x_train, pd.DataFrame):
        raise TypeError("""The 'x_train' parameter must be a pandas DataFrame object""")

    if not isinstance(x_test, pd.DataFrame):
        raise TypeError("""The 'x_test' parameter must be a pandas DataFrame object""")

    if not isinstance(y_train, pd.Series):
        raise TypeError("""The 'y_train' parameter must be a pandas Series""")

    if not isinstance(y_test, pd.Series):
        raise TypeError("""The 'y_test' parameter must be a pandas Series""")   

    if not isinstance(intercept, bool):
        raise ValueError("""The 'intercept' parameter must be logical""")
            
    if penalty not in [None, 'l1']:
        raise ValueError("""The 'penalty' parameter must be iether None or 'l1'; got {}""".format(penalty))
        
    if not isinstance(alpha, float) or not 0 < alpha < 1:
        raise ValueError("""The 'alpha' parameter must be float from 0 to 1; got {}""".format(alpha))

    if check_sample not in ['train', 'test']:
        raise ValueError("""The 'check_sample' parameter must be in ['train', 'test']; got {}""". format(check_sample))
        
    if metric not in ['gini', 'auc']:
        raise ValueError("""The 'metric' parameter must be in ['gini', 'auc']; got {}""".format(metric))

    if not isinstance(p_value, float) or not 0 < p_value < 0.5:
        raise ValueError("""The 'p_value' parameter must be float from 0 to 0.5; got {}""".format(p_value))

    if not isinstance(gini_cutoff, float) or not 0 < gini_cutoff < 1:
        raise ValueError("""The 'gini_cutoff' parameter must be from 0 to 1; got {}""".format(gini_cutoff))

    if not isinstance(auc_cutoff, float) or not 0 < auc_cutoff < 1:
        raise ValueError("""The 'auc_cutoff' parameter must be from 0 to 1; got {}""".format(auc_cutoff))
        
    # =============================================================================
    # Generating models
    # =============================================================================
    items = list(range(len(x_train.columns)))
    
    final_models = {}
    temp = {}
    
    for i in tqdm(range(max_model_number)):
        comb = list(random.sample(items, dependent_number))
        var = [x_train.columns[k] for k in comb ]
        x_train_temp = x_train[var]
        x_test_temp = x_test[var]
        
        try:
            model = LogitModel(x_train = x_train_temp
                               , y_train = y_train
                               , x_test = x_test_temp
                               , y_test = y_test
                               , intercept = intercept
                               , penalty = penalty
                               , alpha = alpha
                               )
            model.Model_SK()
            model.Model_SM()
        except:
            continue
                
        # =============================================================================
        # checking sample       
        # =============================================================================
        if  IsModelValid(model = model
                         , coef_expectation = coef_expectation
                         , p_value = p_value
                         , check_sample = check_sample
                         , metric = metric
                         , gini_cutoff = gini_cutoff
                         , auc_cutoff = auc_cutoff
                         ):
            temp[i] = model
            final_models.update(temp)
                
    return final_models
        

def ModelMetaInfo(
            models_dict: dict
            , sort_by: str
        ) -> pd.DataFrame:
    
    """
    The function provides a meta information on the model obtained
    
    Parameters:
    ----------
        models_dict: dict
            dictionary containing models
                        
        sort_by: str {'gini_test', 'gini_train', 'auc_test', 'auc_train', 'Brier_test', 'Brier_train', 'F1_test', 'F1_train'}
            a column name to sort by
            
    Returns:
    --------
        final_data: pd.DataFrame
            
    """
    
    # =============================================================================
    # Validating parameters
    # =============================================================================
    
    if not isinstance(models_dict, dict):
        raise TypeError("""The 'model_dict' parameter must a dictionary""")
    
    if sort_by not in ['gini_test', 'gini_train'
                       , 'auc_test', 'auc_train'
                       , 'Brier_test', 'Brier_train'
                       , 'F1_test', 'F1_train']:
        raise ValueError("""The 'sort_by' parameter must be in  ['gini_test', 'gini_train', 'auc_test', 'auc_train', 'Brier_test', 'Brier_train', 'F1_test', 'F1_train']; got {}""".format(sort_by))
      
    # =============================================================================
    # Generating Meta Information
    # =============================================================================
    final_data = pd.DataFrame()
    
    for i in models_dict.keys():
        temp = pd.DataFrame(
            {
                'gini_test':[ models_dict[i].Gini_Test() ]
                , 'gini_train': [models_dict[i].Gini_Train()]
                , 'auc_test': [models_dict[i].AUC_Test()]
                , 'auc_train': [models_dict[i].AUC_Train()]
                , 'Brier_test': [models_dict[i].Brier_Test()]
                , 'Brier_train': [models_dict[i].Brier_Train()]
                , 'F1_test': [models_dict[i].F1_Test()]
                , 'F1_train': [models_dict[i].F1_Train()]
                }
            )
        temp.index = [i]
        
        final_data = pd.concat([final_data, temp])
        final_data = final_data.sort_values(sort_by, ascending=False)
        
    return final_data
        

def ModelAggregation(
        models_dict: dict
        , metric: str
        , check_sample: str
        ) -> dict:
    
    """
    The funnction calculates the weigths of each model in the ensemble
    
    Paratemers:
    -----------
        
        models_dict: dict
            dictionary of LogitModel() instances
            
        metric: str {'gini', 'auc', 'f1', 'brier'}
            metric for calculation weighted average
            
        check_sample: str {'test', 'train'}
            sample to calculate metrics for averaging
                        
    Returns:
    --------
        weights_dict: dict
            a dictionary with weights for each model in the ensemble
    """
    
    # =============================================================================
    # Validating parameters  
    # =============================================================================
    if not isinstance(models_dict, dict):
        raise TypeError("""The 'model_dict' parameter must a dictionary""")
    
    if check_sample not in ['train', 'test']:
        raise ValueError("""The 'check_sample' parameter must be in ['train', 'test']; got {}""".format(check_sample))
        
    if metric not in ['gini', 'auc', 'f1', 'brier']:
        raise ValueError("""The 'metric' parameter must be in ['gini', 'auc', 'f1', 'brier']; got {}""".format(metric))
        
    # =============================================================================
    # Accuracy summation    
    # =============================================================================
    accuracy_sum = 0
    if metric == 'auc':
        metric = metric.upper()
    else:
        metric = metric.capitalize()
    method_name = metric + "_" + check_sample.capitalize()
    
    for key in models_dict.keys():
        if metric == 'Brier':
            accuracy_sum += 1/getattr(models_dict[key], method_name)()
        else:
            accuracy_sum += getattr(models_dict[key], method_name)()
                
    weights_dict = {}
    for key in models_dict.keys():
        if metric == 'Brier':
            weights_dict[key] = (1/getattr(models_dict[key], method_name)())/accuracy_sum
        else:
            weights_dict[key] = getattr(models_dict[key], method_name)()/accuracy_sum
        
    return weights_dict
        
        
def PredictionAggregation(
            models_dict: dict
            , weights_dict: dict
            , x_data: pd.DataFrame
        )  ->   pd.DataFrame:
    
    """
    The Function calculates the PD using aggregation scheme. 
    For each model in the ensemble the corresponding weight is assinged.
    
    Parameters:
    ----------
        models_dict: dict
            a dictionary with LogitModel instances
            
        weights_dict: dict
            a dictionary with weights for each model in the models_dict dictionary. The keys of both models_dict and weights_dict must be the same
            
        x_data: pd.DataFrame
            a pandas DataFrame with the explanatory variable to predict the PD
            
    Output:
    -------
        pred: pd.DataFrame
            a pandas DataFrame with clients  PD's
    """      

    # =============================================================================
    # Validating parameters  
    # =============================================================================
    if not isinstance(models_dict, dict):
        raise TypeError("""The 'model_dict' parameter must a dictionary""")

    if set(models_dict.keys()) != set(weights_dict.keys()):
        raise ValueError("The keys of models_dict and weights_dict are not the same")

    # =============================================================================
    # Generating prediction
    # =============================================================================
    y_proba_final = pd.Series()
    
    for key in models_dict.keys():
        temp_proba = pd.Series(models_dict[key].Prediction(x_data, logprob = False)) * weights_dict[key]        
        y_proba_final = pd.concat([y_proba_final, temp_proba], axis = 1)
        
    pred = y_proba_final.sum(axis = 1)
           
    return pred 
   
    
def ModelStacking(
        models_dict: dict
        , x_data: pd.DataFrame
        , y_data: pd.Series
        , penalty: Optional[float] = None
        , alpha: float = 0.5
        , fit_intercept: bool = True
        ) -> LogisticRegression:
    
    """
    The Function the model of sklearn as of stacking shceme
    
    Parameters:
    -----------
        models_dict: dict
            a dictionary with LogitModel instances
            
        x_data: pd.DataFrame
            a pandas DataFrame with explanatory variables 
            
        y_data: pd.Series
            a pandas Series with discrete dependent variable
            
        penalty: str optional {'None', 'l1', 'l2'}, default = None
            a regularization for Logistic Regression model
            
        alpha: float, default = 0.5
            a float variable for the regularization
            
        fit_intercept: bool, {'True', 'False'}, default = 'True'
            a bool variable whether to fit intercept in the Logistic Regression or not         
    
    Returns:
    --------
        model: LogisticRegression
            a LogisticRegression model from sklearn package
    """
 
    # =============================================================================
    # Validating parameters  
    # =============================================================================    
    if not isinstance(models_dict, dict):
        raise TypeError("""The 'model_dict' parameter must a dictionary""")

    if not isinstance(x_data, pd.DataFrame):
        raise TypeError("""The 'x_data' parameter must be a pandas DataFrame object""")
        
    if not isinstance(y_data, pd.Series):
        raise TypeError("""The 'y_data' parameter must be a pandas Series""")
    
    if len(x_data) != len(y_data):
        raise ValueError("""The length of x_data and y_data must be identical""")
        
    if penalty not in [None, 'l1', 'l2']:
        raise ValueError("""The 'penalty' parameter must be in [None, 'l1', 'l2']; got {}""".format(penalty))
    
    if not isinstance(alpha, float) or not 0 < alpha < 1:
        raise ValueError("""The 'alpha' parameter must be float from 0 to 1; got {}""".format(alpha))
            
    if not isinstance(fit_intercept, bool):
        raise TypeError("""The 'fit_intercept' parameter must be logical""")
              
    # =============================================================================
    # Stacking model    
    # =============================================================================
    y_proba = pd.Series()
    
    for key in models_dict.keys():
        temp_proba = pd.Series(models_dict[key].Prediction(x_data = x_data, logprob = False))
        y_proba = pd.concat([y_proba, temp_proba], axis = 1)
        
    y_proba.columns = [i for i in range(len(y_proba.columns))]
    y_proba = y_proba.drop(columns = [0])
    
    model = LogisticRegression(
            penalty = penalty
            , C = alpha
            , fit_intercept = fit_intercept
            , solver='saga'
        ).fit(y_proba, y_data)
    
    return model
        
        
def PredictionStacking(
        models_dict: dict
        , x_data: pd.DataFrame
        , model: ModelStacking
        ) -> pd.Series:
    
    """
    The function calculates the final PD based on the models obtained with stacking scheme
    
    Parameters:
    -----------
    
        models_dict: dict
            a dictionary with LogitModel instances
            
        x_data: pd.DataFrame
            a pandas dataframe with data of clients to calculate PD
            
        model: ModelStacking
            a Logistec Regression model obtain using ModelStacking function
            
    Returns:
    --------
        pred: pd.Series
            a predicted PD for a client using x_data and stacking model
    """
    # =============================================================================
    # Validating parameters  
    # =============================================================================    
    if not isinstance(models_dict, dict):
        raise TypeError("""The 'model_dict' parameter must a dictionary""")
    
    if not isinstance(x_data, pd.DataFrame):
        raise TypeError("""The 'x_data' parameter must be a pandas DataFrame object""")

    if not isinstance(model, LogisticRegression):
        raise TypeError("""The 'model' parameter must be a Logistic object""")
    
    # =============================================================================
    # Stacking Prediction    
    # =============================================================================
    y_proba = pd.Series()
    
    for key in models_dict.keys():
        
        temp_proba = pd.Series(models_dict[key].Prediction(x_data, logprob = False))
        y_proba = pd.concat([y_proba, temp_proba], axis = 1)
    
    y_proba.columns = [i for i in range(len(y_proba.columns))]
    y_proba = y_proba.drop(columns = [0])
    
    pred = pd.Series(model.predict_proba(y_proba)[:,1]).round(4)
    
    pred.index = x_data.index
    
    return pred
          

def AggregationMetaInfo(models_dict: dict
                        , x_train: pd.DataFrame
                        , y_train: pd.Series
                        , x_test: pd.DataFrame
                        , y_test: pd.Series
                        ) -> pd.DataFrame:
    
    """
    The function presents the accuracy results of aggregation and stacking.
    
    Parameters:
    -----------
        models_dict: dict
            a dictionary with LogitModel instances
            
        x_train: pd.DataFrame
            a pandas DataFrame with explanatory variables of training set
            
        y_train: pd.Series
            a pandas Series with discrete dependent variable of training set
            
        x_test: pd.DataFrame
            a pandas DataFrame with explanatory variables of testing set
            
        y_test: pd.Series
            a pandas Series with discrete dependent variable of testing set
        
    Returns:
    --------
        final_data: pd.DataFrame
            a pandas DataFrame with accuracy results of different aggregation methods and stacking sorted by Gini    
    """
    # =============================================================================
    # Validating parameters  
    # =============================================================================    
    if not isinstance(models_dict, dict):
        raise TypeError("""The 'model_dict' parameter must a dictionary""")

    if not isinstance(x_train, pd.DataFrame):
        raise TypeError("""The 'x_train' parameter must be a pandas DataFrame object""")

    if not isinstance(x_test, pd.DataFrame):
        raise TypeError("""The 'x_test' parameter must be a pandas DataFrame object""")

    if not isinstance(y_train, pd.Series):
        raise TypeError("""The 'y_train' parameter must be a pandas Series""")

    if not isinstance(y_test, pd.Series):
        raise TypeError("""The 'y_test' parameter must be a pandas Series""")     

    if len(x_train) != len(y_train):
        raise ValueError("""The length of 'x_tarin' and 'y_train' must be identical""")    
    
    if len(x_test) != len(y_test):
        raise ValueError("""The length of 'x_test' and 'y_test' must be identical""")    
    # =============================================================================
    # Aggregation sector    
    # =============================================================================
    
    final_data = pd.DataFrame()
    for metric in ['gini', 'auc', 'f1', 'brier']:
        for sample in ['test', 'train']:
            if sample == 'test':
                x_data = x_test
                y_data = y_test
            else:
                x_data = x_train
                y_data = y_train
        
            weights = ModelAggregation(models_dict = models_dict
                                       , metric = metric
                                       , check_sample = sample
                                       )
            
            pred = PredictionAggregation(models_dict = models_dict
                                         , weights_dict = weights
                                         , x_data = x_data
                                         )
            auc = roc_auc_score(y_true = y_data, y_score = pred)
            auc = float("{:.3f}".format(auc))  
            
            gini = 2 * auc -1
            gini = float("{:.3f}".format(gini))            
            
            name = metric + "_" + sample
            temp = pd.DataFrame(
                    {'gini': [gini]
                     , 'auc': [auc]}
                    , index = [name]
                )
            
            final_data = pd.concat([final_data, temp])
            
    # =============================================================================
    # Stacking Sector
    # =============================================================================
    for sample in ['test', 'train']:
        if sample == 'test':
            x_data = x_test
            y_data = y_test
        else:
            x_data = x_train
            y_data = y_train
        model_stack = ModelStacking(models_dict = models_dict
                                    , x_data = x_data
                                    , y_data = y_data
                                    )  
    
        pred_stack = PredictionStacking(models_dict = models_dict
                                        , x_data = x_data
                                        , model = model_stack) 
    
        auc_stack = roc_auc_score(y_true = y_data, y_score = pred_stack)
        auc_stack = float("{:.3f}".format(auc_stack))  
        
        gini_stack = 2 * auc_stack -1
        gini_stack = float("{:.3f}".format(gini_stack))
        
        temp = pd.DataFrame(
                {'gini': [gini_stack]
                 , 'auc': [auc_stack]}
                , index = ["stacking"+"_"+sample]
            )   
        final_data = pd.concat([final_data, temp])
    final_data = final_data.sort_values(by = ['gini'], ascending=False)

    return final_data      
            
  
 