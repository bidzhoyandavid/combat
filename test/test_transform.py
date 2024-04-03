# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 12:52:45 2024

@author: bidzh
"""

import optbinning as ob
from optbinning import OptimalBinning, BinningProcess
import pandas as pd
import numpy as np
import pytest

from combat.transform import *

from contextlib import nullcontext as does_not_raise


data  = pd.read_excel('heloc_dataset_v1.xlsx' )
df_sign = pd.read_excel("Sign Expec.xlsx", sheet_name='heloc (2)', index_col='Variable')

data['default'] = data['RiskPerformance'].apply(lambda x: 1 if x in ['Bad'] else 0 )
data = data.drop(columns = ['RiskPerformance'])

y = data['default']
x = data.drop(columns = ['default'])
# variable_names = list(x.columns)

special_codes = [-9, -8, -7]


name = 'ExternalRiskEstimate'
name_1 = 'MSinceOldestTradeOpen'

class Test_Transform():
    
    @pytest.mark.parametrize(
            "x, y, mon_constraint, var_name, var_type, metric, special_codes, solver, max_pvalue, divergence, prebinning_method, max_n_prebins, expectation"
            , [
                (x[name], y, -1,  name, 'numerical', 'woe', special_codes,  'cp', 0.05, 'iv', 'cart', 20, does_not_raise())
                , (x[name_1], y, 1,  name_1, 'numerical', 'woe', special_codes,  'cp', 0.05, 'iv', 'cart', 20,  does_not_raise())
                , (x[name_1], y, 1,  name_1, 'numerical', 'woe', special_codes,  'cp', 0.05, 'js', 'cart', 20, does_not_raise())
                , (x[name_1], y, 1,  name_1, 'numerical', 'woe', special_codes,  'cp', 0.05, 'js', 'mdlp', 20, does_not_raise())
                , (x[name_1], y, 1,  name_1, 'numerical', 'woe', special_codes,  'cp', 0.05, 'js', 'quantile', 20, does_not_raise())
                , (x[name_1], y, 1,  name_1, 'numerical', 'woe', special_codes,  'cp', 0.05, 'js', 'uniform', 20, does_not_raise())
                # , (x[name_1], y, 1,  name_1, 'numerical', 'woe', special_codes,  'ls', 0.05, 'iv', 'uniform', 20, does_not_raise()) # problrm occurs ---------------------
                , (x[name_1], y, 1,  name_1, 'numerical', 'woe', special_codes,  'mip', 0.05, 'js', 'uniform', 20, does_not_raise())
                , (x[name_1], y, 1,  name_1, 'numerical', 'woe', special_codes,  'cp', 0.05, 'js', None, 20, pytest.raises(ValueError))
                , (x[name_1], y, -1,  123, 'numerical', 'woe', special_codes,  'cp', 0.05, 'iv','cart', 20, pytest.raises(TypeError))
                , (x[name_1], y, -1,  name_1, 'numerical1', 'woe', special_codes,  'cp', 0.05, 'iv', 'cart', 20, pytest.raises(ValueError))
                , (x[name_1], y, -1,  name_1, 'numerical', 'woe124', special_codes,  'cp', 0.05, 'iv', 'cart', 20, pytest.raises(ValueError))
                , (x[name_1], y, -1,  name_1, 'numerical', 'woe', special_codes,  'cp123', 0.05, 'iv', 'cart', 20, pytest.raises(ValueError))
                , (x[name_1], y, -1,  name_1, 'numerical', 'woe', special_codes,  'cp', 1.05, 'iv', 'cart', 20, pytest.raises(ValueError))
                , (x[name_1], y, -1,  name_1, 'numerical', 'woe', special_codes,  'cp', '1.05', 'iv', 'cart', 20, pytest.raises(ValueError))
                , (x[name_1], y, -1,  name_1, 'numerical', 'woe', special_codes,  'cp', 0.05, 'ivv', 'cart', 20, pytest.raises(ValueError))
                , (x[name_1], y, -1,  name_1, 'numerical', 'woe', special_codes,  'cp', 0.05, 'iv', 'carttt', 20, pytest.raises(ValueError))
                , (x[name_1], y, -1,  name_1, 'numerical', 'woe', special_codes,  'cp', 0.05, 'iv', 'cart', -20, pytest.raises(ValueError))
                , (x[name_1], y, -1,  name_1, 'numerical', 'woe', special_codes,  'cp', 0.05, 'iv', 'cart', '-20', pytest.raises(ValueError))
            ]
    )
    def test_WOETransform(self, x, y, mon_constraint, var_name, var_type, metric, special_codes, solver, max_pvalue, divergence, prebinning_method, max_n_prebins, expectation):
        with expectation:
            assert isinstance(WoETransform(x=x
                                           , y=y
                                           , mon_constraint = mon_constraint
                                           , var_name = var_name
                                           , var_type = var_type
                                           , metric = metric
                                           , special_codes = special_codes
                                           , solver = solver
                                           , max_pvalue = max_pvalue
                                           , divergence = divergence
                                           , prebinning_method = prebinning_method
                                           , max_n_prebins = max_n_prebins
                                           )
                                           , dict)
            
    @pytest.mark.parametrize(
            "x_data, y_data, df_sign, metric, divergence, prebinning_method, max_n_prebins, min_prebin_size, verbose, min_n_bins, expectation"
            , [
                (x, y, df_sign, 'woe', 'iv', 'cart', 20, 0.4, False, 5, does_not_raise())
                , (x, y, df_sign, 'event_rate', 'iv', 'cart', 20, 0.4, 5, False, does_not_raise())
                , (x, y, df_sign, 'woe', 'js', 'cart', 20, 0.4, False, 5, does_not_raise())
                , (x, y, df_sign, 'woe', 'iv', 'mdlp', 20, 0.4, False, 5, does_not_raise())
                , (x, y, df_sign, 'woe', 'iv', 'uniform', 20, 0.4, False, 5, does_not_raise())
                , (x, y, df_sign, 'woe', 'iv', 'quantile', 20, 0.4, False, 5, does_not_raise())
                , (x, y, df_sign, 'woe', 'iv', 'quantile', 20, 0.4, True, 5, does_not_raise())
                , (x[1:], y, df_sign, 'woe', 'iv', 'quantile', 20, 0.4, False, 5, pytest.raises(ValueError))
                , (x, y, df_sign, 'woead', 'iv', 'quantile', 20, 0.4, False, 5, pytest.raises(ValueError))
                , (x, y, df_sign, 'woe', 'iasdv', 'quantile', 20, 0.4, False, 5, pytest.raises(ValueError))
                , (x, y, df_sign, 'woe', 'iv', 'quantile12', 20, 0.4, False, 5, pytest.raises(ValueError))
                , (x, y, df_sign, 'woe', 'iv', 'quantile', -20, 0.4, False, 5, pytest.raises(ValueError))
                , (x, y, df_sign, 'woe', 'iv', 'quantile', '20', 0.4, False, 5, pytest.raises(ValueError))
                , (x, y, df_sign, 'woe', 'iv', 'quantile', 20, 0.6, False, 5, pytest.raises(ValueError))
                , (x, y, df_sign, 'woe', 'iv', 'quantile', 20, '0.6', False, 5, pytest.raises(ValueError))
                , (x, y, df_sign, 'woe', 'iv', 'quantile', 20, -0.6, False, 5, pytest.raises(ValueError))
                , (x, y, df_sign, 'woe', 'iv', 'quantile', 20, 0.4, 'False', 5, pytest.raises(TypeError))
                , (x, y, df_sign, 'woe', 'iv', 'quantile', 20, 0.4, False, -5, pytest.raises(TypeError))
                , (x, y, df_sign, 'woe', 'iv', 'quantile', 20, 0.4, False, '-5', pytest.raises(TypeError))
                
                
            ]
    )
    def test_WoEDataPreparation(self, x_data, y_data, df_sign, metric,  divergence, prebinning_method, max_n_prebins, min_prebin_size, verbose, min_n_bins, expectation):
        with expectation:
            assert isinstance(WoEDataPreparation(x_data = x_data
                                                 , y_data = y_data
                                                 , df_sign = df_sign
                                                 , metric = metric
                                                 , divergence = divergence
                                                 , prebinning_method = prebinning_method
                                                 , max_n_prebins = max_n_prebins
                                                 , min_prebin_size = min_prebin_size
                                                 , verbose = verbose
                                                 , min_n_bins =min_n_bins
                                                 )
                                                 , dict)












