# -*- coding: utf-8 -*-
"""
The Short_List module is a powerful component of this COMBAT package, designed to streamline data exploration and analysis tasks with its efficient functionalities. 
This module is tailored to provide concise yet robust tools for comparing means across different groups and exploring the variance explained by various factors within datasets.

Functions within the Short_List module:

1. `MeanComparison(x_train_0, x_train_1, equal_var, alternative)` - offers a straightforward solution for comparing means across different groups or categories within datasets

2. `VarExpPower(y_train, x_train, y_test, x_test, discriminatory, vif, individual_accuracy, check_sample)` -  enables users to explore the variables explanatory power 

The Short_List module empowers users to conduct efficient data exploration and analysis tasks by offering concise yet powerful functionalities for comparing means and exploring variance explained within datasets. 
With these tools, users can quickly gain insights into their data, identify patterns, and make informed decisions, ultimately enhancing their analytical workflows.
"""

import pandas as pd
import numpy as np

from scipy.stats import (ttest_ind
                         , kruskal                         
                         )

from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm 

from combat.models import LogitModel
from combat.transform import WoETransform



def MeanComparison(
          x_train_0: pd.Series
        , x_train_1: pd.Series
        , equal_var: bool = False
        , alternative: str = 'two-sided'
        ) -> dict:
    
    """
    The function conduct test to two samples mean comparison. 
    The function implies both parametric t-test and non-parametric Kruskal-Wallis H-test
    
    Parameters:
    -----------
        x_train_0: pd.Series() 
            a pd.Series of a feature of 0 group
            
        x_train_1: pd.Series() 
            a pd.Series of a feature of 2 group
            
        equal_var: bool, optional, default = False
            an indicator of the assumption of equal variance. 
                            
        alternative: str, optiona, {'two-sided', 'less', 'greater'}, default = 'two-sided'
            type of alternative hypothesis
                            
    Returns:
    -------
        final_data: dict 
            keys: 
                ttest 
                kruskal 
            values:
                tt: tuple(statistic, pvalue) - results of conducted t-test 
                krusk: tuple(statistic, pvalue) - results of conducted Kruskal-Wallis H-test               
    """
    
    # =============================================================================
    # Validating parameters    
    # =============================================================================
    if not isinstance(x_train_0, pd.Series):
        raise TypeError("""The 'x_train_0' parameter must be pd.Series""")
        
    if not isinstance(x_train_1, pd.Series):
        raise TypeError("""The 'x_train_1' parameter must be pd.Series""")
        
    if not isinstance(equal_var, bool):
        raise TypeError("""The 'equal_var' parameter must be logical""")
    
    if alternative not in ('two-sided', 'less', 'greater'):
        raise ValueError("""The 'alternative' parameter must be in ('two-sided', 'less', 'greater'); got {}""".format(alternative))
    
    # =============================================================================
    # Mean comparison            
    # =============================================================================
    tt = ttest_ind(a = list(x_train_0)
                  , b = list(x_train_1)
                  , equal_var=equal_var
                  ) 
       
    krusk = kruskal(
             x_train_0
            , x_train_1
        )
    
    final_data = {
            "ttest": tt
            , "kruskal": krusk
        }
            
    return final_data
       

def VarExpPower(
            y_train: pd.Series
            , x_train: pd.DataFrame
            , y_test: pd.Series
            , x_test: pd.DataFrame
            , discriminatory: str = 'ttest' 
            , vif: bool = True
            , individual_accuracy: str = 'gini'
            , check_sample: str = 'test'
        ) -> pd.DataFrame:
    
    """
    The function describes the data in terms of explanatory power
    
    Parameters:
    -----------    
        y_train: pd.Series() 
            the series of binary dependent variable of train set
        
        x_train: pd.DataFrame() 
            the dataframe of all independent variables of train set
        
        y_test: pd.Series() 
            the series of binary dependent variable of test set
        
        x_test: pd.DataFrame() 
            the dataframe of all independent variables of test set
        
        discriminatory: str {'ttest', 'kruskal'}, default = 'ttest' 
            which test to perform to conduct the test to compare samples means
            the default is 'ttest'.
                            
        vif: bool , default = True
            calculate the Variance inflation factor or not
            If False VIFs will not be calculated
                        
        individual_accuracy: str {'gini', 'auc', 'f1_score' }, defualt = 'gini' 
            calculates individual accuracy in pairwise regression.
                                
        check_sample: str {'test', 'train'}, default = 'test'
            data sample to perform individual accuracy test
                
        woe: bool, default = True
            wheter the input data is woe_transformed or not
                                    
    Outputs:
    --------
        final_data: pd.DataFrame() 
                pandas DataFrame with all variables analysis   
    """
    # =============================================================================
    # Validating parameters    
    # =============================================================================
    if not isinstance(x_train, pd.DataFrame):
        raise TypeError("""The 'x_train' parameter must be a pandas DataFrame object""")

    if not isinstance(x_test, pd.DataFrame):
        raise TypeError("""The 'x_test' parameter must be a pandas DataFrame object""")

    if not isinstance(y_train, pd.Series):
        raise TypeError("""The 'y_train' parameter must be a pandas Series""")

    if not isinstance(y_test, pd.Series):
        raise TypeError("""The 'y_test' parameter must be a pandas Series""")        

    if discriminatory not in ('ttest', 'kruskal'):
        raise ValueError("""The 'discriminatory' parameter must be in ('ttest', 'kruskal'); got {}""".format(discriminatory))
        
    if not isinstance(vif, bool):
        raise TypeError("""The 'vif' parameter must be logical""")
        
    if individual_accuracy not in ('gini', 'auc', 'f1_score'):
        raise ValueError("""The 'individual_accuracy' must be in ('gini', 'auc', 'f1_score'); got {}""".format(individual_accuracy))

    if check_sample not in ['train', 'test']:
        raise ValueError("""The 'check_sample' parameter must be in ['train', 'test']; got {}""".format(check_sample))
        
    # =============================================================================
    # Conducting mean comparison tests     
    # =============================================================================
    y_train= y_train.rename('reference')
    data_train = pd.concat([y_train, x_train], axis = 1, ignore_index=False)
    
    data_train_1 = data_train[data_train.reference == 1]
    data_train_0 = data_train[data_train.reference == 0]   
    
    disc = []
    
    if discriminatory == 'ttest':
        for col in x_train.columns:
            tt = MeanComparison(
                      x_train_0 = data_train_0[col]
                    , x_train_1= data_train_1[col]
                )['ttest'][1]
            disc.append(round(tt, 3))
    else:
        for col in x_train.columns:
            krusk = MeanComparison(
                x_train_0 = data_train_0[col]
                , x_train_1 = data_train_1[col]
                )['kruskal'][1]
            disc.append(round(krusk, 3))
        
    
    # =============================================================================
    # Individual Accuracy Test  
    # =============================================================================
    gini = []
    auc = []
    accur = []
    f1_score = []
    
    if check_sample == 'test':
        for col in x_train.columns:
            model = LogitModel(x_train= pd.DataFrame(x_train[col])
                               , y_train = y_train
                               , x_test = pd.DataFrame(x_test[col])
                               , y_test = y_test
                               )
            model.Model_SK()
            model.Model_SM()
            
            gini.append(model.Gini_Test())
            auc.append(model.AUC_Test())
            f1_score.append(model.F1_Test())
    else:
        for col in x_train.columns:
            model = LogitModel(x_train= pd.DataFrame(x_train[col])
                               , y_train = y_train
                               , x_test = pd.DataFrame(x_test[col])
                               , y_test = y_test
                               )
            model.Model_SK()
            model.Model_SM()
            
            gini.append(model.Gini_Train())
            auc.append(model.AUC_Train())
            f1_score.append(model.F1_Train())
            
    # =============================================================================
    # VIFs        
    # =============================================================================
        
    x_train_const = sm.add_constant(x_train)
    vif_dict = {col: variance_inflation_factor(x_train.values, i) for i, col in enumerate(x_train.columns)}
    vif_dict = pd.DataFrame.from_dict(vif_dict, orient='index', columns = ['vif'])
    
    # =============================================================================
    # Final data  
    # =============================================================================
    
    final_data = pd.DataFrame(index = x_train.columns )
    
    # explanatory power -------------------
    if individual_accuracy == 'gini':
        final_data['gini'] = gini
    elif individual_accuracy == 'auc':
        final_data['auc'] = auc
    elif individual_accuracy == 'f1_score':
        final_data['f1_score'] = f1_score
    else:
        final_data['accuracy'] = accur
    
    
    # discriminatory test ------------------
    if discriminatory == 'ttest':
        final_data['ttest-pvalue'] =  disc
    else:
        final_data['kruskal-pvalue'] = disc

    # vifs ----------------------------------
    if vif:
        final_data['vif'] = vif_dict['vif']      
    
    return final_data
        

    
    
    
    
    
    
    
    
    
    
    
    
