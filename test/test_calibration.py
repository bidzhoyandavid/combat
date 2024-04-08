# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 12:51:48 2024

@author: bidzh
"""

from combat.models import LogitModel
from combat.short_list import *
from combat.combat import *
from combat.transform import *
from combat.calibration import *
from combat.utilities import *

import pytest
from contextlib import nullcontext as does_not_raise


import pandas as pd
import numpy as np

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


# df_expec = df_sign.reset_index().drop(columns = 'dtype')
# df_expec['Variable'] = df_expec['Variable'].apply(lambda x: "woe_" + x)
# df_expec['Expec'] = 0

vars_to_remove = ['NumTrades60Ever2DerogPubRec', 'NumTrades90Ever2DerogPubRec', 'NumInqLast6M', 'NumInqLast6Mexcl7days']

del_data = DeleteVars(
    x_train = x_train
    , x_test = x_test
    , df_sign = df_sign
    , vars_to_remove = vars_to_remove
)

x_train = del_data['x_train_new']
x_test = del_data['x_test_new']
df_sign = del_data['df_sign_new']


model = LogitModel(x_train=x_train
                   , y_train=y_train
                   , x_test = x_test
                   , y_test = y_test)

model.Model_SK()
model.Model_SM()

proba = (model.Predict_Proba_Test())
x_proba = pd.DataFrame(proba[:,1])



calib_model = CalibrationModel(x_proba, y_test)



class Test_Calibration():

    @pytest.mark.parametrize(
        "true_labels, probabilities, n_bins, expectation"
        , [
            (y_test, proba, 20,  does_not_raise())
            , (y_test, proba, -2,  pytest.raises(ValueError))
            , (y_test, proba, '20',  pytest.raises(ValueError))
        ]
    )
    def test_ExpectedCalibrationError(self, true_labels, probabilities, n_bins,  expectation):
        with expectation:
            assert isinstance(ExpectedCalibrationError(true_labels, probabilities, n_bins),  float)

    @pytest.mark.parametrize(
            "x_data, y_data, penalty, alpha, fit_intercept, expectation"
            , [
                (x_test, y_test, None, 0.5, True, does_not_raise())
                , (x_test, y_test, None, 0.5, False, does_not_raise())
                , (x_test, y_test, 'l1', 0.5, True, does_not_raise())
                , (x_test, y_test, 'l2', 0.5, True, does_not_raise())
                , (x_test, y_test, 'l1', 0.5, False, does_not_raise())
                , (x_test, y_test, 'l2', 0.5, False, does_not_raise())
                , (x_test, y_test, 'l1', '0.5', False, pytest.raises(ValueError))
                , (x_test, y_test, 'l1', 1.5, False, pytest.raises(ValueError))
                , (x_test, y_test, 'l1', -1.5, False, pytest.raises(ValueError))
                , (x_test, y_test, 'l113', 0.5, False, pytest.raises(ValueError))
                , (x_test, y_test[1:], 'l1', 0.5, False, pytest.raises(ValueError))
            ]
    )
    def test_CalibrationModel(self, x_data, y_data, penalty, alpha, fit_intercept, expectation):
        with expectation:
            assert isinstance(CalibrationModel(x_data, y_data, penalty, alpha, fit_intercept), LogisticRegression)

    @pytest.mark.parametrize(
            "x_data, model, logprob, expectation"
            , [
                (x_proba, calib_model, False, does_not_raise())
                , (x_proba, calib_model, True, does_not_raise())
                , (x_proba, calib_model, [True], pytest.raises(TypeError))
                , (x_proba, calib_model, 'True', pytest.raises(TypeError))
            ]
    )
    def test_PredictionCalibration(self, x_data, model, logprob, expectation):
        with expectation:
            assert isinstance(PredictionCalibration(x_data, model, logprob), pd.Series)

    @pytest.mark.parametrize(
            "y_data, probabilities, n_bins, label, expectation"
            , [
                (y_test, proba[:,1], 20, 'aaa', does_not_raise())
                , (y_test, proba[:,1], -20, 'aaa', pytest.raises(ValueError))
                , (y_test, proba[:,1], '-20', 'aaa', pytest.raises(ValueError))
                , (y_test, proba[:,1], 0, 'aaa', pytest.raises(ValueError))
                , (y_test, proba[:,1], 10, 12, pytest.raises(ValueError))
            ]
    )
    def test_CalibrationCurve(self, y_data, probabilities, n_bins, label, expectation):
        with expectation:
            with plt.ioff():
                CalibrationCurve(y_data, probabilities, n_bins, label)
                plt.close()
                assert True
                                                                                            
