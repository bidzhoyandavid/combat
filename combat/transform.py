# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 12:56:01 2024

@author: bidzh
"""

import optbinning as ob
from optbinning import OptimalBinning, BinningProcess
import pandas as pd


def WoETransform(
            x: pd.Series
            , y: pd.Series
            , mon_constraint: int
            , special_codes: list
            , var_name: str
            , var_type: str
            , metric: str
            , min_n_bins: int
            , solver: str = 'cp'
        ) -> dict:
    """
    The function perdorms the WoE transformation

    Parameters:
    -----------
        x: pd.Series
            a pandas Series of explanatory variable
            
        mon_constraint: int {-1, 0, 1}
            numeric type of monotonic constraint
            
        special_codes: list
            special codes in the data 
            
        var_name: str
            a varibale name
        
        y: pd.Series
            a pandas Series of target binary variable
            
        var_type: str {'numerical', 'categorical'}
            a type of explanatory variable
            
        metric: str {'woe', 'event_rate'}
            a metric to perform transformation
            
        solver: str {'cp', 'mip', 'ls'}, default = 'cp'
            a solver method to perform optimal binning
            
    Returns:
    -------
        final_data: dict
            a dictionary with transformed data, status, plot and binning table
    """
    
    # =============================================================================
    # Validating parameters    
    # =============================================================================
    if not isinstance(x, pd.Series):
        raise TypeError("""The 'x' paramater must a pd.Series""")

    if not isinstance(y, pd.Series):
        raise TypeError("""The 'y' paramater must a pd.Series""")

    if var_type not in ['numerical', 'categorical']:
        raise ValueError("There is no {} varibale type".format(var_type))
        
    if metric not in ['woe', 'event_rate']:
        raise ValueError("There is no {} metric".format(metric))
        
    if mon_constraint not in [-1, 0, 1]:
        raise ValueError("There is no {} mon_constraint".format(mon_constraint))
        
    # =============================================================================
    # Optimal binning    
    # =============================================================================    
    if mon_constraint == -1:
        monotonic_trend = 'descending'
    elif mon_constraint == 1:
        monotonic_trend = 'ascending'
    else:
        monotonic_trend = 'auto'
    
    optb = OptimalBinning(name = var_name
                          , dtype = var_type
                          , solver = solver
                          , monotonic_trend  = monotonic_trend
                          , min_n_bins = min_n_bins
                          )
    
    optb.fit(x, y)
    
    x_transform = optb.transform(x, metric = metric)
    x_transform = pd.DataFrame(x_transform)
    x_transform.columns = ['woe_'+var_name]
    
    final_data = {
        "status": optb.status
        , 'binning_table': optb.binning_table.build()
        # , "plot": optb.binning_table.plot(metric = metric)
        , 'woe_transform': x_transform        
        }
    
    return final_data



def WoEDataPreparation(
        x_data: pd.DataFrame
        , y_data: pd.Series
        , df_sign: pd.DataFrame
        , special_codes: list
        , metric: str 
        , min_n_bins: int
        , solver: str = 'cp'
        ) -> pd.DataFrame:
    
    # =============================================================================
    # Validating parameters    
    # =============================================================================
    # if len(df_sign.columns) != 3:
    #     raise ValueError("""The 'df_sign' must be a pd.DataFrame with 3 columns:
    #                          name: the name of the variable
    #                          dtype: type of the variable. {'numerical', 'categorical'}
    #                          expec: sign expectation in (-1, 0, 1)"""
    #                          )
    
    # df_sign.columns = ['name', 'dtype', 'expec']
    
    final_data = {}
    bin_table = {}
    x_woe = pd.DataFrame()
    status = pd.DataFrame()
    optimizer = ('mip', 'ls')
    for col in x_data.columns:
        temp = WoETransform(x = pd.Series(x_data[col].values)
                            , y = y_data
                            , mon_constraint = df_sign.loc[col, 'Expec']
                            , special_codes = special_codes
                            , var_name = col 
                            , var_type = df_sign.loc[col, 'dtype']
                            , metric = 'woe'
                            , min_n_bins=min_n_bins
                            )
        # in case of non OPTIMAL status of binning run binning with other optimizer
        k = 0
        while temp['status'] != 'OPTIMAL' and k<=1:
            temp = WoETransform(x = pd.Series(x_data[col].values)
                                , y = y_data
                                , mon_constraint = df_sign.loc[col, 'Expec']
                                , special_codes = special_codes
                                , var_name = col 
                                , var_type = df_sign.loc[col, 'dtype']
                                , metric = 'woe'
                                , min_n_bins=min_n_bins
                                , solver=optimizer[k]
                                )
            k += 1
            
            
        temp_status = pd.DataFrame({'name': col
                                    , 'status': temp['status']
                                    , 'iv': temp['binning_table'].loc['Totals', 'IV'].round(4)
                                    , 'js': temp['binning_table'].loc['Totals', 'JS'].round(4)
                                    }, index = [0]) 
        bin_table[col] = temp['binning_table']
        
        status = pd.concat([status, temp_status], axis = 0)           
        x_woe = pd.concat([x_woe, temp['woe_transform']], axis = 1)
        
    status['name'] = status['name'].apply(lambda x: 'woe_' + x)
        
    final_data['status'] = status
    final_data['x_woe'] = x_woe
    final_data['bining_tables'] = bin_table
        
    return final_data
        
        
        
        
        
        