# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 12:52:23 2024

@author: bidzh
"""


from combat.models import LogitModel
from combat.short_list import *
from combat.combat import *
from combat.transform import *
from combat.calibration import *
from combat.scorecard import *
import pytest
from contextlib import nullcontext as does_not_raise


import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import Union

import random 
from random import randrange
from itertools import combinations

from sklearn.model_selection import train_test_split
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


model = LogitModel(x_train=x_train
                   , y_train=y_train
                   , x_test = x_test
                   , y_test = y_test)

model.Model_SK()
model.Model_SM()

proba = (model.Predict_Proba_Test())

class Test_ScoreCard():

    @pytest.mark.parametrize(
            "y_proba, log, target_score, target_odds, pdo, expectation"
            , [
                (proba[:,1], True, 600, 30, 20, does_not_raise())
                , (proba[:,1], False, 600, 30, 20, does_not_raise())
                , (proba[:,1], True, 200, 30, 20, does_not_raise())
                , (proba[:,1], True, 600, 50, 20, does_not_raise())
                , (proba[:,1], True, 600, 30, 50, does_not_raise())
                , (proba[:,1], 'True', 600, 30, 20, pytest.raises(TypeError))
                , (proba[:,1], True, '600', 30, 20, pytest.raises(TypeError))
                , (proba[:,1], True, 600, '30', 20, pytest.raises(TypeError))
                , (proba[:,1], True, 600, 30, '20', pytest.raises(TypeError))
                , (proba[:,1], True, -600, 30, 20, pytest.raises(ValueError))
                , (proba[:,1], True, 600, -30, 20, pytest.raises(ValueError))
                , (proba[:,1], True, 600, 30, -20, pytest.raises(ValueError))
            ]
    )
    def test_Scorescrd(self, y_proba, log, target_score, target_odds, pdo, expectation):
        with expectation:
            assert isinstance(ScoreCard(y_proba, log, target_score, target_odds, pdo), pd.DataFrame)