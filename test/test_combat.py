# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 12:51:59 2024

@author: bidzh
"""

from combat.combat import *

import pandas as pd
import numpy as np
import pytest

from combat.short_list import *
from combat.transform import *
from combat.models import LogitModel 
from combat.utilities import *


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


model_comb_1 = ModelCombination(y_train = y_train
                              , x_train = x_train
                              , y_test = y_test
                              , x_test = x_test
                              , max_model_number = 1000
                              , dependent_number = 5
                              , coef_expectation = df_sign
                              , gini_cutoff=0.4
                              , p_value = 0.1
                              , intercept = True
                              , penalty = None
                              )



aggr_weight = ModelAggregation(models_dict=model_comb_1, check_sample='test', metric='gini')

aggr_pred = PredictionAggregation(models_dict=model_comb_1, weights_dict=aggr_weight, x_data = x_test)

stack_model = ModelStacking(model_comb_1, x_test, y_test)
x_train_1 = x_train[1:]
x_test_1 = x_test[1:]


class Test_Combat():
    
    @pytest.mark.parametrize(
        "coef_exp, p_value, check_sample, metric, gini_cutoff, auc_cutoff, expectation"
        , [
            (df_sign, 0.05, 'test', 'gini', 0.5, 0.75,  does_not_raise())
            , (df_sign, 0.05, 'train', 'gini', 0.5, 0.75,  does_not_raise())
            , (df_sign, 0.05, 'test', 'auc', 0.5, 0.75,  does_not_raise())
            , (df_sign, 0.05, 'train', 'auc', 0.6, 0.75,  does_not_raise()) 
            , (df_sign, 0.05, 'test', 'ginii', 0.5, 0.75,  pytest.raises(ValueError))
            , (df_sign, 0.05, 'testt', 'auc', 0.5, 0.75,  pytest.raises(ValueError))
            , (df_sign, 0.05, 'test', 'auc', '0.5', 0.75,  pytest.raises(ValueError))
            , (df_sign, 0.05, 'test', 'auc', 0.5, '0.75',  pytest.raises(ValueError))
            , (df_sign, 0.05, 'test', 'auc', 1.5, 0.75,  pytest.raises(ValueError))
            , (df_sign, 0.05, 'test', 'auc', 0.5, 1.75,  pytest.raises(ValueError))
            , (df_sign, '0.05', 'test', 'auc', 0.5, 0.75,  pytest.raises(ValueError))
            , (df_sign, 1.05, 'test', 'auc', 0.5, 0.75,  pytest.raises(ValueError))
            , (df_sign.to_numpy(), 0.05, 'test', 'auc', 0.5, 0.75,  pytest.raises(TypeError))
        ]
    )
    def test_IsModelValid(self, coef_exp, p_value, check_sample, metric, gini_cutoff, auc_cutoff, expectation):
        with expectation:
            model = LogitModel(x_train, y_train, x_test, y_test)
            model.Model_SK()
            model.Model_SM()
            assert isinstance(IsModelValid(model, coef_exp, p_value, check_sample, metric, gini_cutoff, auc_cutoff), bool)
        

    @pytest.mark.parametrize(
        "y_train, x_train, y_test, x_test, max_model_number, dependent_number, coef_expectation, intercept, penalty, alpha, p_value, check_sample, metric, gini_cutoff, auc_cutoff, expectation"
        , [
            ( y_train, x_train, y_test, x_test, 100, 12, df_sign, True, None, 0.5, 0.05, 'test', 'gini', 0.5, 0.75, does_not_raise())
            , ( y_train, x_train, y_test, x_test, 100, 12, df_sign, False, None, 0.5, 0.05, 'test', 'gini', 0.5, 0.75, does_not_raise())
            , ( y_train, x_train, y_test, x_test, 100, 12, df_sign, False, 'l1', 0.5, 0.05, 'test', 'gini', 0.5, 0.75, does_not_raise())
            , ( y_train, x_train, y_test, x_test, 100, 12, df_sign, True, None, 0.5, 0.05, 'train', 'gini', 0.5, 0.75, does_not_raise())
            , ( y_train, x_train, y_test, x_test, 100, 12, df_sign, False, None, 0.5, 0.1, 'test', 'auc', 0.5, 0.75, does_not_raise())

            , ( y_train, x_train, y_test, x_test, 100, 35, df_sign, True, None, 0.5, 0.05, 'test', 'gini', 0.5, 0.75, pytest.raises(ValueError))
            , ( y_train, x_train, y_test, x_test, 100, 12, df_sign, True, 'l2', 0.5, 0.05, 'test', 'gini', 0.5, 0.75, pytest.raises(ValueError))
            , ( y_train, x_train, y_test, x_test, 100, 12, df_sign, 'True', 'l1', 0.5, 0.05, 'test', 'gini', 0.5, 0.75, pytest.raises(ValueError))
            , ( y_train, x_train, y_test, x_test, 100, 12, df_sign, [True], 'l1', 0.5, 0.05, 'test', 'gini', 0.5, 0.75, pytest.raises(ValueError))
            , ( y_train, x_train, y_test, x_test, 100, 12, df_sign, True, None, 1.5, 0.05, 'test', 'gini', 0.5, 0.75, pytest.raises(ValueError))
            , ( y_train, x_train, y_test, x_test, 100, 12, df_sign, True, None, '1.5', 0.05, 'test', 'gini', 0.5, 0.75, pytest.raises(ValueError))
            , ( y_train, x_train, y_test, x_test, 100, 12, df_sign, True, None, 0.5, 1.05, 'test', 'gini', 0.5, 0.75, pytest.raises(ValueError))
            , ( y_train, x_train, y_test, x_test, 100, 12, df_sign, True, None, 0.5, '0.05', 'test', 'gini', 0.5, 0.75, pytest.raises(ValueError))
            , ( y_train, x_train, y_test, x_test, 100, 12, df_sign, True, None, 0.5, 0.05, 'test', 'gini', '0.5', 0.75, pytest.raises(ValueError))
            , ( y_train, x_train, y_test, x_test, 100, 12, df_sign, True, None, 0.5, 0.05, 'test', 'gini', 0.5, '0.75', pytest.raises(ValueError))
            , ( y_train, x_train, y_test, x_test, 100, 12, df_sign, True, None, 0.5, 0.05, 'testt', 'gini', 0.5, 0.75, pytest.raises(ValueError))
            , ( y_train, x_train, y_test, x_test, 100, 12, df_sign, True, None, 0.5, 0.05, 'test', 'sdgini', 0.5, 0.75, pytest.raises(ValueError))
        ]
    )
    def test_ModelCombination(self, y_train, x_train, y_test, x_test, max_model_number, dependent_number, coef_expectation, intercept, penalty, alpha, p_value, check_sample, metric, gini_cutoff, auc_cutoff, expectation):
        with expectation:
            assert isinstance(ModelCombination( y_train, x_train, y_test, x_test,  max_model_number, dependent_number, coef_expectation, intercept, penalty, alpha, p_value, check_sample, metric, gini_cutoff, auc_cutoff), dict)

    @pytest.mark.parametrize(
            "model_dict, sort_by, expectation"
            , [
                (model_comb_1, 'gini_test', does_not_raise())
                ,  (model_comb_1, 'gini_train', does_not_raise())
                ,  (model_comb_1, 'auc_test', does_not_raise())
                ,  (model_comb_1, 'auc_train', does_not_raise())
                ,  (model_comb_1, 'Brier_test', does_not_raise())
                ,  (model_comb_1, 'Brier_train', does_not_raise())
                ,  (model_comb_1, 'F1_test', does_not_raise())
                ,  (model_comb_1, 'F1_train', does_not_raise())
                ,  (model_comb_1, 'gini_trainnnn', pytest.raises(ValueError))
            ]
    )
    def test_ModelMetaInfo(self, model_dict, sort_by, expectation):
        with expectation:
            assert isinstance(ModelMetaInfo(model_dict, sort_by), pd.DataFrame)

    @pytest.mark.parametrize(
            "model_dict, metric, check_sample, expectation"
            , [
                (model_comb_1, 'gini', 'test', does_not_raise())
                , (model_comb_1, 'gini', 'train', does_not_raise())
                , (model_comb_1, 'auc', 'train', does_not_raise())
                , (model_comb_1, 'auc', 'test', does_not_raise())
                , (model_comb_1, 'f1', 'train', does_not_raise())
                , (model_comb_1, 'f1', 'test', does_not_raise())
                , (model_comb_1, 'brier', 'test', does_not_raise())
                , (model_comb_1, 'brier', 'test', does_not_raise())
                , (model_comb_1, 'f11', 'test', pytest.raises(ValueError))
                , (model_comb_1, 'f1', 'teste', pytest.raises(ValueError))
            ]
    )
    def test_ModelAggregation(self, model_dict, metric, check_sample, expectation):
        with expectation:
            assert isinstance(ModelAggregation(model_dict, metric, check_sample), dict)

    @pytest.mark.parametrize(
            "models_dict, weights_dict, x_data, expectation"
            , [
                (model_comb_1, aggr_weight, x_test, does_not_raise())
            ]
    )
    def test_PredictionAggregation(self, models_dict, weights_dict, x_data, expectation):
        with expectation:
            assert isinstance(PredictionAggregation(models_dict, weights_dict, x_data), pd.Series)

    @pytest.mark.parametrize(
            "models_dict, x_data, y_data, penalty, alpha, fit_intercept, expectation"
            , [
                (model_comb_1, x_test, y_test, None, 0.5, True, does_not_raise())
                , (model_comb_1, x_test, y_test, None, 0.5, False, does_not_raise())
                , (model_comb_1, x_test, y_test, 'l1', 0.5, True, does_not_raise())
                , (model_comb_1, x_test, y_test, 'l2', 0.5, True, does_not_raise())
                , (model_comb_1, x_test, y_test, None, None, True, pytest.raises(ValueError))
                , (model_comb_1, x_test, y_test, None, 1.4, True, pytest.raises(ValueError))
                , (model_comb_1, x_test, y_test, None, '1.5', True, pytest.raises(ValueError))
                , (model_comb_1, x_test, y_test, 'l3', None, True, pytest.raises(ValueError))
                , (model_comb_1, x_test, y_test, 'l1', None, 'True', pytest.raises(ValueError))
                , (model_comb_1, x_test, y_test, 'l1', None, [True], pytest.raises(ValueError))
            ]
    )
    def test_ModelStacking(self, models_dict, x_data, y_data, penalty, alpha, fit_intercept, expectation):
        with expectation:
            assert isinstance(ModelStacking(models_dict, x_data, y_data, penalty, alpha, fit_intercept), LogisticRegression)

    @pytest.mark.parametrize(
        "models_dict, x_data, model, expectation"
        , [
            (model_comb_1, x_test, stack_model, does_not_raise())
        ]
    )
    def test_PredictionStacking(self, models_dict, x_data, model, expectation):
        with expectation:
            assert isinstance(PredictionStacking(models_dict, x_data, model), np.ndarray)
            assert len(PredictionStacking(models_dict, x_data, model)) == len(x_test)

    @pytest.mark.parametrize(
            "model_comb_1, x_train, y_train, x_test, y_test, expectation"
            , [
                (model_comb_1, x_train, y_train, x_test, y_test, does_not_raise())
                , (model_comb_1, x_train_1, y_train, x_test, y_test, pytest.raises(ValueError))
                , (model_comb_1, x_train, y_train, x_test_1, y_test, pytest.raises(ValueError))
            ]
    )
    def test_AggregationMetaInfo(self, model_comb_1, x_train, y_train, x_test, y_test, expectation):
        with expectation:
            assert isinstance(AggregationMetaInfo(model_comb_1, x_train, y_train, x_test, y_test), pd.DataFrame)












