# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 12:52:08 2024

@author: bidzh
"""

        
import pandas as pd
import numpy as np
import pytest
from sklearn.model_selection import train_test_split
from typing import Union


from combat.short_list import *
from combat.transform import *
from combat.models import LogitModel 
from combat.utilities import *

from contextlib import nullcontext as does_not_raise


# Test data
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
                                
                                )
    
x_train, x_test, y_train, y_test = train_test_split(final_data['x_woe'], y, test_size=0.2, random_state=42, shuffle=True)

# x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=42, shuffle=True)
 

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


x_train_1 = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], "C": [6, 7, 8]})
y_train_1 = np.array([0, 1, 0])
x_test_1 = pd.DataFrame({'A': [4, 5, 6], 'B': [7, 8, 9], "D": [6, 7, 8]})
y_test_1 = pd.Series([1, 0, 1])

class TestLogitModel:
    @pytest.mark.parametrize(
            "x_train, y_train, x_test, y_test, intercept, penalty, alpha, expectation"
            , [
                (x_train, y_train, x_test, y_test, True, None, 0.5, does_not_raise())
                , (x_train, y_train, x_test, y_test, False, None, 0.5, does_not_raise())
                , (x_train, y_train, x_test, y_test, True, 'l1', 0.5, does_not_raise())
                , (x_train, y_train, x_test, y_test, 'True', 'l1', 0.5, pytest.raises(ValueError))
                , (x_train, y_train, x_test, y_test, True, 'l2', 0.5, pytest.raises(ValueError))
                , (x_train, y_train, x_test, y_test, True, 'l1', '0.5', pytest.raises(ValueError))
                , (x_train, y_train, x_test, y_test, True, 0.8, 0.5, pytest.raises(ValueError))
                , (x_train, y_train, x_test, y_test, True, None, 1.5, pytest.raises(ValueError))
                , (x_train_1, y_train, x_test, y_test, True, None, 0.5, pytest.raises(ValueError))
                , (x_train, y_train, x_test_1, y_test, True, None, 0.5, pytest.raises(ValueError))
                , (x_train, y_train_1, x_test_1, y_test, True, None, 0.5, pytest.raises(TypeError))
                , (x_train, y_train, x_test, y_test, [True], 'l1', 0.5,  pytest.raises(ValueError))
            ]
    )
    def test_logit_model_all_models(self, x_train, y_train, x_test, y_test, intercept, penalty, alpha, expectation):
        with expectation:
            model = LogitModel(x_train, y_train, x_test, y_test, intercept, penalty, alpha)
            assert model.Model_SK() is not None
            assert model.Model_SM() is not None


    def test_Coefs(self, x_train = x_train, y_train = y_train, x_test= x_test, y_test = y_test):
        model = LogitModel(x_train, y_train, x_test, y_test)
        model.Model_SK()
        model.Model_SM()
        assert isinstance(model.Gini_Train(), float)
        assert isinstance(model.AUC_Train(), float)
        assert isinstance(model.Predict_Proba_Train(), np.ndarray)
        assert isinstance(model.Predict_LogProba_Train(), np.ndarray)
        assert isinstance(model.Brier_Train(), float)
        assert isinstance(model.F1_Train(), float)
        assert isinstance(model.FPR_Train(), np.ndarray)
        assert isinstance(model.TPR_Train(), np.ndarray)
        assert isinstance(model.Gini_Test(), float)
        assert isinstance(model.AUC_Test(), float)
        assert isinstance(model.Predict_Proba_Test(), np.ndarray)
        assert isinstance(model.Predict_LogProba_Test(), np.ndarray)
        assert isinstance(model.Brier_Test(), float)
        assert isinstance(model.F1_Test(), float)
        assert isinstance(model.FPR_Test(), np.ndarray)
        assert isinstance(model.TPR_Test(), np.ndarray)


    @pytest.mark.parametrize(
            "x_train, y_train, x_test, y_test, cutoff, expectation"
            , [
                (x_train, y_train, x_test, y_test, 0.1, does_not_raise())
                , (x_train, y_train, x_test, y_test, 1.1, pytest.raises(ValueError))
                , (x_train, y_train, x_test, y_test, -0.1, pytest.raises(ValueError))
                , (x_train, y_train, x_test, y_test, '0.1', pytest.raises(ValueError))
                , (x_train, y_train, x_test, y_test, None, pytest.raises(ValueError))
            ]
    )
    def test_coefs_cutoff(self, x_train , y_train, x_test, y_test, cutoff, expectation):
        with expectation:
            model = LogitModel(x_train, y_train, x_test, y_test)
            model.Model_SK()
            model.Model_SM()
            assert isinstance(model.Accuracy_Train(cutoff), float)
            assert isinstance(model.PredictLabel_Train(cutoff), list)
            assert isinstance(model.Recall_Train(cutoff), float)
            assert isinstance(model.Precision_Train(cutoff), float)
            assert isinstance(model.Confusion_Matrix_Train(cutoff), np.ndarray)
            assert isinstance(model.Accuracy_Test(cutoff), float)
            assert isinstance(model.PredictLabel_Test(cutoff), list)
            assert isinstance(model.Recall_Test(cutoff), float)
            assert isinstance(model.Precision_Test(cutoff), float)
            assert isinstance(model.Confusion_Matrix_Test(cutoff), np.ndarray)


    @pytest.mark.parametrize(
            " x_train , y_train, x_test, y_test, intercept, expectation"
            , [
                ( x_train , y_train, x_test, y_test, True, does_not_raise())
                , ( x_train , y_train, x_test, y_test, False, does_not_raise())
                , ( x_train , y_train, x_test, y_test, 'True', pytest.raises(ValueError))
            ]
    )
    def test_model_results(self,  x_train , y_train, x_test, y_test, intercept, expectation):
        with expectation:
            model = LogitModel(x_train, y_train, x_test, y_test, intercept)
            model.Model_SK()
            model.Model_SM()
            assert isinstance(model.GetIntercept_SK(), np.ndarray)
            assert isinstance(model.GetCoefficients_SK(), np.ndarray)
            assert isinstance(model.GetCoefficients_SM(), pd.DataFrame)


    @pytest.mark.parametrize(
            "x_train , y_train, x_test, y_test, logrob, x_data, expectation"
            , [
                (x_train, y_train, x_test, y_test, True, x_test,  does_not_raise())
                , (x_train, y_train, x_test, y_test, False, x_test, does_not_raise())
                , (x_train, y_train, x_test, y_test, 'True', x_test,  pytest.raises(ValueError))
                , (x_train, y_train, x_test, y_test, [True], x_test, pytest.raises(ValueError))
                , (x_train, y_train, x_test, y_test, [True], np.array(x_test), pytest.raises(TypeError))

            ]
    )
    def test_model_prediction(self,  x_train , y_train, x_test, y_test, logrob, x_data, expectation):
        with expectation:
            model = LogitModel(x_train, y_train, x_test, y_test)
            model.Model_SK()
            model.Model_SM()
            assert isinstance(model.Prediction(x_data, logprob=logrob), np.ndarray)

            
