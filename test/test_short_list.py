# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 12:52:34 2024

@author: bidzh
"""

from combat.combat import *

import pandas as pd
import numpy as np
import pytest

from combat.short_list import *
from combat.transform import *
from combat.models import LogitModel 
from contextlib import nullcontext as does_not_raise

from sklearn.model_selection import train_test_split


data  = pd.read_excel('heloc_dataset_v1.xlsx' )
df_sign = pd.read_excel("Sign Expec.xlsx", sheet_name='heloc (2)', index_col='Variable')

data['default'] = data['RiskPerformance'].apply(lambda x: 1 if x in ['Bad'] else 0 )
data = data.drop(columns = ['RiskPerformance'])

y = data['default']
x = data.drop(columns = ['default'])
# variable_names = list(x.columns)

special_codes = [-9, -8, -7]

# =============================================================================
# woe transformation
# =============================================================================
 
final_data = WoEDataPreparation(x_data = x
                                , y_data = y
                                , df_sign = df_sign
                                , special_codes = special_codes
                                , metric = 'woe'
                                , min_n_bins=1
                                )
    
x_train, x_test, y_train, y_test = train_test_split(final_data['x_woe'], y, test_size=0.2, random_state=42, shuffle=True)


df_expec = df_sign.reset_index().drop(columns = 'dtype')
df_expec['Variable'] = df_expec['Variable'].apply(lambda x: "woe_" + x)
df_expec['Expec'] = 0


x_train = x_train.drop(columns = ['woe_NumTrades60Ever2DerogPubRec', 'woe_NumTrades90Ever2DerogPubRec', 'woe_NumInqLast6M', 'woe_NumInqLast6Mexcl7days'])
x_test = x_test.drop(columns = ['woe_NumTrades60Ever2DerogPubRec', 'woe_NumTrades90Ever2DerogPubRec', 'woe_NumInqLast6M', 'woe_NumInqLast6Mexcl7days'])

df_expec_1 = df_expec.copy()
df_expec = df_expec.drop([5, 6, 15, 16])
y_train= y_train.rename('reference')
x_all = pd.concat([y_train, x_train], axis = 1, ignore_index=False)
data_train_1 = x_all[x_all.reference == 1]
data_train_0 = x_all[x_all.reference == 0]


class Test_Short_List():
    @pytest.mark.parametrize(
            "x_train_0, x_train1, equal_var, alternative, expectation"
            , [
                (data_train_1['woe_ExternalRiskEstimate'], data_train_0['woe_ExternalRiskEstimate'], True, 'two-sided', does_not_raise())
                
            ]
    )
    def test_MeanComparison(self, x_train_0, x_train1, equal_var, alternative, expectation):
        with expectation:
            assert isinstance(MeanComparison(x_train_0, x_train1, equal_var, alternative), dict)
    
    @pytest.mark.parametrize(
        "y_train, x_train, y_test, x_test, discriminatory, vif, individual_accuracy, check_sample, expectation"
        , [
            (y_train, x_train, y_test, x_test, 'ttest', True, 'gini', 'test', does_not_raise())
            , (y_train, x_train, y_test, x_test, 'ttest', False, 'gini', 'test', does_not_raise())
            , (y_train, x_train, y_test, x_test, 'kruskal', False, 'gini', 'test', does_not_raise())
            , (y_train, x_train, y_test, x_test, 'kruskal', False, 'auc', 'test', does_not_raise())
            , (y_train, x_train, y_test, x_test, 'kruskal', False, 'f1_score', 'test', does_not_raise())
            , (y_train, x_train, y_test, x_test, 'ttest', False, 'auc', 'test', does_not_raise())
            , (y_train, x_train, y_test, x_test, 'kruskal', False, 'gini', 'train', does_not_raise())
            , (y_train, x_train, y_test, x_test, 'kruskal', True, 'gini', 'test', does_not_raise())
            , (y_train, x_train, y_test, x_test, 'ttest', False, 'gini', 'train', does_not_raise())

            , (y_train, x_train, y_test, x_test, 'ttes12t', False, 'gini', 'train', pytest.raises(ValueError))   
            , (y_train, x_train, y_test, x_test, 'ttest', False, 'gini12', 'train', pytest.raises(ValueError))
            , (y_train, x_train, y_test, x_test, 'ttest', False, 'gini', 'trainfe', pytest.raises(ValueError)) 
            , (y_train, x_train, y_test, x_test, 'ttest', 'False', 'gini', 'train', pytest.raises(TypeError))        
            , (pd.DataFrame(y_train), x_train, y_test, x_test, 'ttest', False, 'gini', 'train', pytest.raises(TypeError))        
        ]        
    )
    def test_VarExpPower(self, y_train, x_train, y_test, x_test, discriminatory, vif, individual_accuracy, check_sample, expectation):
        with expectation:
            assert isinstance(VarExpPower(y_train, x_train, y_test, x_test, discriminatory, vif, individual_accuracy, check_sample), pd.DataFrame)
            assert len(VarExpPower(y_train, x_train, y_test, x_test, discriminatory, vif, individual_accuracy, check_sample)) == len(x_train.columns)
