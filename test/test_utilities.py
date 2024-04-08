import pandas as pd
from sklearn.model_selection import train_test_split
import pytest
from contextlib import nullcontext as does_not_raise

from combat.combat import *
from combat.utilities import *
from combat.transform import *




data = pd.read_excel('heloc_dataset_v1.xlsx')
df_sign = pd.read_excel("Sign Expec.xlsx", sheet_name='heloc (2)', index_col='Variable')

data['default'] = data['RiskPerformance'].apply(lambda x: 1 if x in ['Bad'] else 0 )
data = data.drop(columns = ['RiskPerformance'])

y = data['default']
x = data.drop(columns = ['default'])

special_codes = [-9, -8, -7]

final_data = WoEDataPreparation(x_data = x
                                , y_data = y
                                , df_sign = df_sign
                                , special_codes = special_codes
                                , metric = 'woe'
                                , min_n_bins=1
                                )

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, shuffle=True)



vars_to_remove = ['NumTrades60Ever2DerogPubRec', 'NumTrades90Ever2DerogPubRec', 'NumInqLast6M', 'NumInqLast6Mexcl7days']

vars_to_remove1 = ['xuz', 'NumTrades60Ever2DerogPubRec', 'NumTrades90Ever2DerogPubRec', 'NumInqLast6M', 'NumInqLast6Mexcl7days']



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

meta = ModelMetaInfo(models_dict = model_comb_1
                     , sort_by = 'gini_test'
                     )



class Test_Utilities():

    @pytest.mark.parametrize(
            "x_train, x_test, df_sign, vars_to_remove, expectation"
            , [
                (x_train, x_test, df_sign, vars_to_remove, does_not_raise())
                , (x_train, x_test, df_sign[1:], vars_to_remove, pytest.raises(ValueError))
                , (x_train, x_test, df_sign, vars_to_remove1, pytest.raises(ValueError))
            ]
            
            )
    def test_DeleteVars(self, x_train, x_test, df_sign, vars_to_remove, expectation):
        with expectation:
            assert isinstance(DeleteVars(x_train, x_test, df_sign, vars_to_remove), dict)
            assert isinstance(DeleteVars(x_train, x_test, df_sign, vars_to_remove)['x_train_new'], pd.DataFrame)
            assert isinstance(DeleteVars(x_train, x_test, df_sign, vars_to_remove)['x_test_new'], pd.DataFrame)
            assert isinstance(DeleteVars(x_train, x_test, df_sign, vars_to_remove)['df_sign_new'], pd.DataFrame)


    @pytest.mark.parametrize(
            "models_dict, meta_data, select_by, cutoff, expectation"
            , [
                (model_comb_1, meta, 'gini_test', 0.5, does_not_raise())
                , (model_comb_1, meta, 'Brier_test', 0.2, does_not_raise())
                , (model_comb_1, meta, 'dfafa', 0.4, pytest.raises(ValueError))
                , (model_comb_1, meta, 'gini_test', '0.4', pytest.raises(ValueError))
                , (model_comb_1, meta, 'gini_test', 1.4, pytest.raises(ValueError))

            ]
    )
    def test_SelectModels(self, models_dict, meta_data, select_by, cutoff, expectation):
        with expectation:
            assert isinstance(SelectModels(models_dict, meta_data, select_by, cutoff), dict)






