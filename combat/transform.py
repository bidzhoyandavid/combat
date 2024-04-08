# -*- coding: utf-8 -*-
"""
The Transform module is a vital component of this COMBAT package, providing essential functionalities for data transformation and preparation, with a particular focus on Weight of Evidence (WoE) encoding techniques. 
Equipped with two key functions, WoETransform and WoEDataPreparation, this module empowers users to preprocess and encode categorical variables effectively, enhancing the performance and interpretability of predictive models.

Functions within the Transform module:

1. `WoETransform(x, y, mon_constraint, var_name, var_type ...)` - facilitates the transformation of categorical variables using Weight of Evidence (WoE) encoding.

2. `WoEDataPreparation(x_data, y_data, df_sign, metric, ...)` - streamlines the data preparation process by applying WoE encoding to categorical and numerical variables and preparing the dataset for model training


The Transform module serves as a valuable resource for users seeking to preprocess and encode categorical variables effectively, particularly in the context of credit scoring, risk modeling, and other predictive modeling tasks. 
With its robust functionalities for WoE encoding and data preparation, this module facilitates the creation of informative and reliable predictive models, ultimately enhancing decision-making processes in various domains.
"""

import optbinning as ob
from optbinning import OptimalBinning, BinningProcess
import pandas as pd
import numpy as np
from typing import Optional, Union

def WoETransform(
            x: pd.Series
            , y: pd.Series
            , mon_constraint: int
            , var_name: str
            , var_type: str
            , metric: str = 'woe'
            , solver: str = 'cp'
            , divergence: str = 'iv' 
            , prebinning_method: str = 'cart'
            , max_n_prebins: int = 20
            , min_prebin_size: float = 0.05
            , min_n_bins: int = None
            , max_n_bins: int = None
            , min_bin_size: float = None
            , max_bin_size: float = None 
            , min_bin_n_nonevent: int = None 
            , max_bin_n_nonevent: int = None
            , min_bin_n_event: int = None 
            , max_bin_n_event: int = None 
            , min_event_rate_diff: float = 0.0 
            , max_pvalue: float = None
            , max_pvalue_policy: str = 'consecutive'
            , gamma: float = 0.0
            , outlier_detector: str = None
            , outlier_params: dict = None
            , class_weight: Union[dict, str] = None
            , cat_cutoff: float = None
            , cat_unknown: Union[float, str] = None
            , user_splits: list = None
            , user_splits_fixed: list = None
            , special_codes: list = None
            , split_digits: int = None
            , mip_solver: str = 'bop'
            , time_limit: int = 100
            , verbose: bool = False
        ) -> dict:
    """
    The function perdorms the WoE transformation

    Parameters:
    -----------
        x: pd.Series
            a pandas Series of explanatory variable
            
        y: pd.Series
            a pandas Series of target binary variable

        mon_constraint: int {-1, 0, 1}
            numeric type of monotonic constraint
            
        special_codes: list
            special codes in the data 
            
        var_name: str
            a varibale name
            
        var_type: str {'numerical', 'categorical'}
            a type of explanatory variable
            
        metric: str, {'woe', 'event_rate'}, default = 'woe'
            a metric to perform transformation
            
        prebinning_method : str, {'cart', 'mdlp', 'quantile', 'uniform', None}, default="cart"
            the pre-binning method. 
 
        solver : str, {'cp', 'mip', 'ls'}, default="cp"
            the optimizer to solve the optimal binning problem. 
            
        divergence : str, {'iv', 'js', 'hellinger', 'triangular'}, default="iv"
            the divergence measure in the objective function to be maximized.
    
        max_n_prebins : int, default=20
            the maximum number of bins after pre-binning (prebins).
    
        min_prebin_size : float, default=0.05
            the fraction of mininum number of records for each prebin.
    
        min_n_bins : int or None, optional, default=None
            The minimum number of bins. If None, then ``min_n_bins`` is
            a value in ``[0, max_n_prebins]``.
    
        max_n_bins : int or None, optional, default=None
            the maximum number of bins. If None, then ``max_n_bins`` is
            a value in ``[0, max_n_prebins]``.
    
        min_bin_size : float or None, optional, default=None
            the fraction of minimum number of records for each bin. If None,
            ``min_bin_size = min_prebin_size``.
    
        max_bin_size : float or None, optional, default=None
            the fraction of maximum number of records for each bin. If None,
            ``max_bin_size = 1.0``.
    
        min_bin_n_nonevent : int or None, optional, default=None
            the minimum number of non-event records for each bin. If None,
            ``min_bin_n_nonevent = 1``.
    
        max_bin_n_nonevent : int or None, optional, default=None
            the maximum number of non-event records for each bin. If None, then an
            unlimited number of non-event records for each bin.
    
        min_bin_n_event : int or None, optional, default=None
            the minimum number of event records for each bin. If None,
            ``min_bin_n_event = 1``.
    
        max_bin_n_event : int or None, optional, default=None
            the maximum number of event records for each bin. If None, then an
            unlimited number of event records for each bin.
    
        min_event_rate_diff : float, default=0
            the minimum event rate difference between consecutives bins. For solver
            "ls", this option currently only applies when monotonic_trend is
            “ascending”, “descending”, “peak_heuristic” or “valley_heuristic”.
    
        max_pvalue : float or None, optional, default=None
            the maximum p-value among bins. The Z-test is used to detect bins
            not satisfying the p-value constraint. Option supported by solvers
            "cp" and "mip".
    
        max_pvalue_policy : str, default="consecutive"
            the method to determine bins not satisfying the p-value constraint.
            Supported methods are "consecutive" to compare consecutive bins and
            "all" to compare all bins.
    
        gamma : float, default=0
            regularization strength to reduce the number of dominating bins. Larger
            values specify stronger regularization. Option supported by solvers
            "cp" and "mip".
    
        outlier_detector : str or None, optional, default=None
            the outlier detection method. Supported methods are "range" to use
            the interquartile range based method or "zcore" to use the modified
            Z-score method.
    
        outlier_params : dict or None, optional, default=None
            dictionary of parameters to pass to the outlier detection method.
    
        class_weight : dict, "balanced" or None, optional (default=None)
            weights associated with classes in the form ``{class_label: weight}``.
            If None, all classes are supposed to have weight one.
    
        cat_cutoff : float or None, optional, default=None
            generate bin others with categories in which the fraction of
            occurrences is below the  ``cat_cutoff`` value. This option is
            available when ``dtype`` is "categorical".
    
        cat_unknown : float, str or None, default=None
            the assigned value to the unobserved categories in training but
            occurring during transform.
    
            If None, the assigned value to an unknown category follows this rule:
    
               - if transform metric == 'woe' then woe(mean event rate) = 0
               - if transform metric == 'event_rate' then mean event rate
               - if transform metric == 'indices' then -1
               - if transform metric == 'bins' then 'unknown'
    
        user_splits : array-like or None, optional, default=None
            the list of pre-binning split points when ``var_type`` is "numerical" or
            the list of prebins when ``var_type`` is "categorical".
    
        user_splits_fixed : array-like or None, default=None
            the list of pre-binning split points that must be fixed.
    
        special_codes : array-like, dict or None, optional, default=None
            list of special codes. Use special codes to specify the data values
            that must be treated separately.
    
        split_digits : int or None, optional, default=None
            the significant digits of the split points. If ``split_digits`` is set
            to 0, the split points are integers. If None, then all significant
            digits in the split points are considered.
    
        mip_solver : str, {'bop', 'cbc'}, default="bop"
            the mixed-integer programming solver. Supported solvers are "bop" to
            choose the Google OR-Tools binary optimizer or "cbc" to choose the
            COIN-OR Branch-and-Cut solver CBC.
    
        time_limit : int, default=100
            the maximum time in seconds to run the optimization solver.
    
        verbose : bool, default=False
            enable verbose output.
            
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

    if mon_constraint not in (-1, 0, 1):
        raise ValueError("""The 'mon_constraint' paramater must be in (-1, 0, 1); got {}""".format(mon_constraint))
    
    if not isinstance(var_name, str):
        raise TypeError("""The 'var_name' paramater must be a string""")

    if var_type not in ("categorical", "numerical"):
        raise ValueError("""The 'var_type' paramater must be in ("categorical", "numerical"): got {}""".format(var_type))
        
    if metric not in('woe', 'event_rate'):
        raise ValueError("""The 'metric' parameter must be in ('woe', 'event_rate'); got {}""".format(metric))

    if prebinning_method not in ("cart", "mdlp", "quantile", "uniform"):
        raise ValueError("""The 'prebinning_method' paramater must be in ("cart", "mdlp", "quantile", "uniform"); got {}""".format(prebinning_method))

    if solver not in ("cp", "ls", "mip"):
        raise ValueError("""The 'solver' paramater must be in ("cp", "ls", "mip"); got {}""".format(solver))

    if divergence not in ("iv", "js", "hellinger", "triangular"):
        raise ValueError("""The 'divergence' paramater must be in ("iv", "js", "hellinger", "triangular"); got {}""".format(divergence))

    if not isinstance(max_n_prebins, int) or max_n_prebins <= 1:
        raise ValueError("The 'max_prebins' paramater must be an integer greater than 1; got {}.".format(max_n_prebins))

    if not isinstance(min_prebin_size, float) or not 0. < min_prebin_size <= 0.5:
        raise ValueError("""The 'min_prebin_size' paramater must be in (0, 0.5]; got {}""".format(min_prebin_size))

    if min_n_bins is not None:
        if not isinstance(min_n_bins, int) or min_n_bins <= 0:
            raise ValueError("The 'min_n_bins' paramater must be a positive integer; got {}".format(min_n_bins))

    if max_n_bins is not None:
        if not isinstance(max_n_bins, int) or max_n_bins <= 0:
            raise ValueError("The 'max_n_bins' paramater must be a positive integer; got {}".format(max_n_bins))

    if min_n_bins is not None and max_n_bins is not None:
        if min_n_bins > max_n_bins:
            raise ValueError("The 'min_n_bins' paramater must be <= 'max_n_bins'; got {} <= {}.".format(min_n_bins, max_n_bins))

    if min_bin_size is not None:
        if (not isinstance(min_bin_size, int) or not 0. < min_bin_size <= 0.5):
            raise ValueError("The 'min_bin_size' paramater must be in (0, 0.5]; got {}.".format(min_bin_size))

    if max_bin_size is not None:
        if (not isinstance(max_bin_size, int) or not 0. < max_bin_size <= 1.0):
            raise ValueError("The 'max_bin_size' paramater must be in (0, 1.0]; got {}.".format(max_bin_size))

    if min_bin_size is not None and max_bin_size is not None:
        if min_bin_size > max_bin_size:
            raise ValueError("The 'min_bin_size' paramater must be <= 'max_bin_size'; got {} <= {}".format(min_bin_size, max_bin_size))

    if min_bin_n_nonevent is not None:
        if (not isinstance(min_bin_n_nonevent, int) or min_bin_n_nonevent <= 0):
            raise ValueError("The 'min_bin_n_nonevent' paramater must be a positive integer; got {}".format(min_bin_n_nonevent))

    if max_bin_n_nonevent is not None:
        if (not isinstance(max_bin_n_nonevent, int) or max_bin_n_nonevent <= 0):
            raise ValueError("The 'max_bin_n_nonevent' paramater must be a positive integer; got {}".format(max_bin_n_nonevent))

    if min_bin_n_nonevent is not None and max_bin_n_nonevent is not None:
        if min_bin_n_nonevent > max_bin_n_nonevent:
            raise ValueError("The 'min_bin_n_nonevent' paramater must be <= max_bin_n_nonevent; got {} <= {}".format(min_bin_n_nonevent, max_bin_n_nonevent))

    if min_bin_n_event is not None:
        if (not isinstance(min_bin_n_event, int) or min_bin_n_event <= 0):
            raise ValueError("The 'min_bin_n_event' paramater must be a positive integer; got {}".format(min_bin_n_event))

    if max_bin_n_event is not None:
        if (not isinstance(max_bin_n_event, int) or max_bin_n_event <= 0):
            raise ValueError("The 'max_bin_n_event' paramater must be a positive integer; got {}".format(max_bin_n_event))

    if min_bin_n_event is not None and max_bin_n_event is not None:
        if min_bin_n_event > max_bin_n_event:
            raise ValueError("The 'min_bin_n_event' paramater must be <= max_bin_n_event; got {} <= {}".format(min_bin_n_event, max_bin_n_event))

    if (not isinstance(min_event_rate_diff, float) or not 0. <= min_event_rate_diff <= 1.0):
        raise ValueError("The 'min_event_rate_diff' paramater must be in [0, 1]; got {}".format(min_event_rate_diff))

    if max_pvalue is not None:
        if (not isinstance(max_pvalue, float) or not 0. < max_pvalue <= 1.0):
            raise ValueError("The 'max_pvalue' paramater must be in (0, 1.0]; got {}.".format(max_pvalue))

    if max_pvalue_policy not in ("all", "consecutive"):
        raise ValueError("The 'max_pvalue_policy' paramater must be in ('all', 'consecutive'); got {}".format(max_pvalue_policy))

    if not isinstance(gamma, float) or gamma < 0:
        raise ValueError("The 'gamma' paramater must be float and >= 0; got {}.".format(gamma))

    if outlier_detector is not None:
        if outlier_detector not in ("range", "zscore"):
            raise ValueError("""The 'outlier_detector' paramater must be in ("range", "zscore"); got {}""".format(outlier_detector))

        if outlier_params is not None:
            if not isinstance(outlier_params, dict):
                raise TypeError("""The 'outlier_params' paramater must be dict; got {}""".format(outlier_params))

    if class_weight is not None:
        if not isinstance(class_weight, (dict, str)):
            raise TypeError("""The 'class_weight' paramater must be  dict, "balanced" or None; got {}""".format(class_weight))
                
        elif isinstance(class_weight, str) and class_weight != "balanced":
            raise ValueError("""The 'class_weight' paramater must be 'balanced'""")

    if cat_cutoff is not None:
        if (not isinstance(cat_cutoff, float) or not 0. < cat_cutoff <= 1.0):
            raise ValueError("The 'cat_cutoff' paramater must be in (0, 1.0]; got {}.".format(cat_cutoff))

    if cat_unknown is not None:
        if not isinstance(cat_unknown, (float, str)):
            raise TypeError("The 'cat_unknown' paramater must be a float or string, depending on the metric used in transform.")

    if user_splits is not None:
        if not isinstance(user_splits, (np.ndarray, list)):
            raise TypeError("The 'user_splits' paramater must be a list or numpy.ndarray.")

    if user_splits_fixed is not None:
        if user_splits is None:
            raise ValueError("The 'user_splits' paramater must be provided.")
        else:
            if not isinstance(user_splits_fixed, (np.ndarray, list)):
                raise TypeError("The 'user_splits_fixed' paramater must be a list or numpy.ndarray.")
            elif not all(isinstance(s, bool) for s in user_splits_fixed):
                raise ValueError("The 'user_splits_fixed' paramater must be list of boolean.")
            elif len(user_splits) != len(user_splits_fixed):
                raise ValueError("Inconsistent length of 'user_splits' and 'user_splits_fixed' paramaters: {} != {}. Lengths must be equal".format(len(user_splits), len(user_splits_fixed)))

    if special_codes is not None:
        if not isinstance(special_codes, (np.ndarray, list, dict)):
            raise TypeError("The 'special_codes' paramater must be a dict, list or numpy.ndarray.")

        if isinstance(special_codes, dict) and not len(special_codes):
            raise ValueError("The 'special_codes' paramater is empty. The 'special_codes' dict must contain at least one special.")

    if split_digits is not None:
        if (not isinstance(split_digits, int) or not 0 <= split_digits <= 8):
            raise ValueError("The 'split_digits' paramater must be an integer in [0, 8]; got {}.".format(split_digits))

    if mip_solver not in ("bop", "cbc"):
        raise ValueError("""The 'mip_solver' paramater must be in ("bop", "cbc"); got {}""".format(mip_solver))

    if not isinstance(time_limit, int) or time_limit < 0:
        raise ValueError("The 'time_limit' paramater must be a positive value in seconds; got {}.".format(time_limit))

    if not isinstance(verbose, bool):
        raise TypeError("The 'verbose' paramater must be a boolean; got {}.".format(verbose))
         
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
                          , divergence = divergence
                          , prebinning_method = prebinning_method
                          , max_n_prebins = max_n_prebins
                          , min_prebin_size = min_prebin_size
                          , min_n_bins = min_n_bins
                          , max_n_bins = max_n_bins
                          , min_bin_size = min_bin_size
                          , max_bin_size = max_bin_size
                          , min_bin_n_nonevent = min_bin_n_nonevent
                          , max_bin_n_nonevent = max_bin_n_nonevent
                          , min_bin_n_event = min_bin_n_event
                          , max_bin_n_event = max_bin_n_event
                          , min_event_rate_diff = min_event_rate_diff
                          , max_pvalue = max_pvalue
                          , max_pvalue_policy = max_pvalue_policy
                          , gamma = gamma
                          , outlier_detector = outlier_detector
                          , outlier_params = outlier_params
                          , class_weight = class_weight
                          , cat_cutoff = cat_cutoff
                          , cat_unknown = cat_unknown
                          , user_splits = user_splits
                          , user_splits_fixed = user_splits_fixed
                          , special_codes = special_codes
                          , split_digits = split_digits
                          , mip_solver = mip_solver
                          , time_limit = time_limit
                          , verbose = verbose
                          )
    
    optb.fit(x, y)
    
    x_transform = optb.transform(x, metric = metric)
    x_transform = pd.DataFrame(x_transform)
    # x_transform.columns = [metric+"_"+var_name]
    x_transform.columns = [var_name]
    
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
        , metric: str = 'woe'
        , divergence: str = 'iv' 
        , prebinning_method: str = 'cart'
        , max_n_prebins: int = 20
        , min_prebin_size: float = 0.05
        , min_n_bins: Optional[int] = None
        , max_n_bins: Optional[int] = None
        , min_bin_size: Optional[float] = None
        , max_bin_size: Optional[float] = None 
        , min_bin_n_nonevent: Optional[int] = None 
        , max_bin_n_nonevent: Optional[int] = None
        , min_bin_n_event: Optional[int] = None 
        , max_bin_n_event: Optional[int] = None 
        , min_event_rate_diff: float = 0.0 
        , max_pvalue: float = None
        , max_pvalue_policy: Optional[str] = 'consecutive'
        , gamma: Optional[float] = 0.0
        , outlier_detector: str = None
        , outlier_params: dict = None
        , class_weight: Optional[Union[dict, str]] = None
        , cat_cutoff: Optional[float] = None
        , cat_unknown: Optional[Union[float, str]] = None
        , user_splits: Optional[list] = None
        , user_splits_fixed: Optional[list] = None
        , special_codes: Optional[list] = None
        , split_digits: Optional[int] = None
        , mip_solver: str = 'bop'
        , time_limit: int = 100
        , verbose: bool = False
        ) -> pd.DataFrame:
    """
    The function prepares the WOE-transformed data

    Parameters:
    -----------
        x_data: pd.DataFrame
            a pandas DataFrame of explanatory variable
            
        y_data: pd.Series
            a pandas Series of target binary variable

        df_sign: pd.DataFrame
            a pandas DataFrame with sign expectations
                                                
        metric: str  {'woe', 'event_rate'}, default = 'woe'
            a metric to perform transformation

        divergence : str {'iv', 'js', 'hellinger', 'triangular'}, default="iv"
            the divergence measure in the objective function to be maximized.
    
        prebinning_method : str {'cart', 'mdlp', 'quantile', 'uniform', None}, default="cart"
            the pre-binning method. 
    
        max_n_prebins : int default=20
            the maximum number of bins after pre-binning (prebins).
    
        min_prebin_size : float default=0.05
            the fraction of mininum number of records for each prebin.
    
        min_n_bins : int or None, optional, default=None
            The minimum number of bins. If None, then ``min_n_bins`` is
            a value in ``[0, max_n_prebins]``.
    
        max_n_bins : int or None, optional, default=None
            the maximum number of bins. If None, then ``max_n_bins`` is
            a value in ``[0, max_n_prebins]``.
    
        min_bin_size : float or None, optional, default=None
            the fraction of minimum number of records for each bin. If None,
            ``min_bin_size = min_prebin_size``.
    
        max_bin_size : float or None, optional, default=None
            the fraction of maximum number of records for each bin. If None,
            ``max_bin_size = 1.0``.
    
        min_bin_n_nonevent : int or None, optional, default=None
            the minimum number of non-event records for each bin. If None,
            ``min_bin_n_nonevent = 1``.
    
        max_bin_n_nonevent : int or None, optional, default=None
            the maximum number of non-event records for each bin. If None, then an
            unlimited number of non-event records for each bin.
    
        min_bin_n_event : int or None, optional, default=None
            the minimum number of event records for each bin. If None,
            ``min_bin_n_event = 1``.
    
        max_bin_n_event : int or None, optional, default=None
            the maximum number of event records for each bin. If None, then an
            unlimited number of event records for each bin.
    
        min_event_rate_diff : float, default=0
            the minimum event rate difference between consecutives bins. For solver
            "ls", this option currently only applies when monotonic_trend is
            “ascending”, “descending”, “peak_heuristic” or “valley_heuristic”.
    
        max_pvalue : float or None, optional, default=None
            the maximum p-value among bins. The Z-test is used to detect bins
            not satisfying the p-value constraint. Option supported by solvers
            "cp" and "mip".
    
        max_pvalue_policy : str,  default="consecutive"
            the method to determine bins not satisfying the p-value constraint.
            Supported methods are "consecutive" to compare consecutive bins and
            "all" to compare all bins.
    
        gamma : float,  default=0
            regularization strength to reduce the number of dominating bins. Larger
            values specify stronger regularization. Option supported by solvers
            "cp" and "mip".
    
        outlier_detector : str or None, optional, default=None
            the outlier detection method. Supported methods are "range" to use
            the interquartile range based method or "zcore" to use the modified
            Z-score method.
    
        outlier_params : dict or None, optional, default=None
            dictionary of parameters to pass to the outlier detection method.
    
        class_weight : dict, "balanced" or None, optional (default=None)
            weights associated with classes in the form ``{class_label: weight}``.
            If None, all classes are supposed to have weight one.
    
        cat_cutoff : float or None, optional, default=None
            generate bin others with categories in which the fraction of
            occurrences is below the  ``cat_cutoff`` value. This option is
            available when ``dtype`` is "categorical".
    
        cat_unknown : float, str or None, optional default=None
            the assigned value to the unobserved categories in training but
            occurring during transform.
    
            If None, the assigned value to an unknown category follows this rule:
    
               - if transform metric == 'woe' then woe(mean event rate) = 0
               - if transform metric == 'event_rate' then mean event rate
               - if transform metric == 'indices' then -1
               - if transform metric == 'bins' then 'unknown'
    
        user_splits : array-like or None, optional, default=None
            the list of pre-binning split points when ``var_type`` is "numerical" or
            the list of prebins when ``var_type`` is "categorical".
    
        user_splits_fixed : array-like or None, optional, default=None
            the list of pre-binning split points that must be fixed.

        special_codes: list or None, optional, default = None
            list of special codes. Use special codes to specify the data values
            that must be treated separately. 
    
        split_digits : int or None, optional, default=None
            the significant digits of the split points. If ``split_digits`` is set
            to 0, the split points are integers. If None, then all significant
            digits in the split points are considered.
    
        mip_solver : str, {'bop', 'cbc'}, default="bop"
            the mixed-integer programming solver. Supported solvers are "bop" to
            choose the Google OR-Tools binary optimizer or "cbc" to choose the
            COIN-OR Branch-and-Cut solver CBC.
    
        time_limit : int, default=100
            the maximum time in seconds to run the optimization solver.
    
        verbose : bool, default=False
            enable verbose output.
            
    Returns:
    -------
        final_data: dict
            a dictionary with transformed data, status, plot and binning table

    """
    # =============================================================================
    # Validating parameters    
    # =============================================================================
    if not isinstance(x_data, pd.DataFrame):
        raise ValueError("""The 'x_data' must be a pandas DataFrame""")
    
    if not isinstance(y_data, pd.Series):
        raise ValueError("""The 'y_data' must be a pandas Series""")
    
    if len(x_data) != len(y_data):
        raise ValueError("""The length of 'x_data' and 'y_data' must be identical""")
                        
    if metric not in('woe', 'event_rate'):
        raise ValueError("""The 'metric' parameter must be in ('woe', 'event_rate'); got {}""".format(metric))

    if divergence not in ("iv", "js", "hellinger", "triangular"):
        raise ValueError("""The 'divergence' paramater must be in ("iv", "js", "hellinger", "triangular"); got {}""".format(divergence))
 
    if prebinning_method not in ("cart", "mdlp", "quantile", "uniform"):
        raise ValueError("""The 'prebinning_method' paramater must be in ("cart", "mdlp", "quantile", "uniform"); got {}""".format(prebinning_method))

    if not isinstance(max_n_prebins, int) or max_n_prebins <= 1:
        raise ValueError("The 'max_prebins' paramater must be an integer greater than 1; got {}".format(max_n_prebins))

    if not isinstance(min_prebin_size, float) or not 0. < min_prebin_size <= 0.5:
        raise ValueError("""The 'min_prebin_size' paramater must be in (0, 0.5]; got {}""".format(min_prebin_size))

    if min_n_bins is not None:
        if not isinstance(min_n_bins, int) or min_n_bins <= 0:
            raise ValueError("The 'min_n_bins' paramater must be a positive integer; got {}".format(min_n_bins))

    if max_n_bins is not None:
        if not isinstance(max_n_bins, int) or max_n_bins <= 0:
            raise ValueError("The 'max_n_bins' paramater must be a positive integer; got {}".format(max_n_bins))

    if min_n_bins is not None and max_n_bins is not None:
        if min_n_bins > max_n_bins:
            raise ValueError("The 'min_n_bins' paramater must be <= 'max_n_bins'; got {} <= {}.".format(min_n_bins, max_n_bins))

    if min_bin_size is not None:
        if (not isinstance(min_bin_size, int) or not 0. < min_bin_size <= 0.5):
            raise ValueError("The 'min_bin_size' paramater must be in (0, 0.5]; got {}.".format(min_bin_size))

    if max_bin_size is not None:
        if (not isinstance(max_bin_size, int) or not 0. < max_bin_size <= 1.0):
            raise ValueError("The 'max_bin_size' paramater must be in (0, 1.0]; got {}.".format(max_bin_size))

    if min_bin_size is not None and max_bin_size is not None:
        if min_bin_size > max_bin_size:
            raise ValueError("The 'min_bin_size' paramater must be <= 'max_bin_size'; got {} <= {}".format(min_bin_size, max_bin_size))

    if min_bin_n_nonevent is not None:
        if (not isinstance(min_bin_n_nonevent, int) or min_bin_n_nonevent <= 0):
            raise ValueError("The 'min_bin_n_nonevent' paramater must be a positive integer; got {}".format(min_bin_n_nonevent))

    if max_bin_n_nonevent is not None:
        if (not isinstance(max_bin_n_nonevent, int) or max_bin_n_nonevent <= 0):
            raise ValueError("The 'max_bin_n_nonevent' paramater must be a positive integer; got {}".format(max_bin_n_nonevent))

    if min_bin_n_nonevent is not None and max_bin_n_nonevent is not None:
        if min_bin_n_nonevent > max_bin_n_nonevent:
            raise ValueError("The 'min_bin_n_nonevent' paramater must be <= max_bin_n_nonevent; got {} <= {}".format(min_bin_n_nonevent, max_bin_n_nonevent))

    if min_bin_n_event is not None:
        if (not isinstance(min_bin_n_event, int) or min_bin_n_event <= 0):
            raise ValueError("The 'min_bin_n_event' paramater must be a positive integer; got {}".format(min_bin_n_event))

    if max_bin_n_event is not None:
        if (not isinstance(max_bin_n_event, int) or max_bin_n_event <= 0):
            raise ValueError("The 'max_bin_n_event' paramater must be a positive integer; got {}".format(max_bin_n_event))

    if min_bin_n_event is not None and max_bin_n_event is not None:
        if min_bin_n_event > max_bin_n_event:
            raise ValueError("The 'min_bin_n_event' paramater must be <= max_bin_n_event; got {} <= {}".format(min_bin_n_event, max_bin_n_event))

    if (not isinstance(min_event_rate_diff, float) or not 0. <= min_event_rate_diff <= 1.0):
        raise ValueError("The 'min_event_rate_diff' paramater must be in [0, 1]; got {}".format(min_event_rate_diff))

    if max_pvalue is not None:
        if (not isinstance(max_pvalue, float) or not 0. < max_pvalue <= 1.0):
            raise ValueError("The 'max_pvalue' paramater must be in (0, 1.0]; got {}".format(max_pvalue))

    if max_pvalue_policy not in ("all", "consecutive"):
        raise ValueError("The 'max_pvalue_policy' paramater must be in ('all', 'consecutive'); got {}".format(max_pvalue_policy))

    if not isinstance(gamma, float) or gamma < 0:
        raise ValueError("The 'gamma' paramater must be float and >= 0; got {}".format(gamma))

    if outlier_detector is not None:
        if outlier_detector not in ("range", "zscore"):
            raise ValueError("""The 'outlier_detector' paramater must be in ("range", "zscore"); got {}""".format(outlier_detector))

        if outlier_params is not None:
            if not isinstance(outlier_params, dict):
                raise TypeError("""The 'outlier_params' paramater must be dict; got {}""".format(outlier_params))

    if class_weight is not None:
        if not isinstance(class_weight, (dict, str)):
            raise TypeError("""The 'class_weight' paramater must be  dict, "balanced" or None; got {}""".format(class_weight))
                
        elif isinstance(class_weight, str) and class_weight != "balanced":
            raise ValueError("""The 'class_weight' paramater must be 'balanced'""")

    if cat_cutoff is not None:
        if (not isinstance(cat_cutoff, float) or not 0. < cat_cutoff <= 1.0):
            raise ValueError("The 'cat_cutoff' paramater must be in (0, 1.0]; got {}".format(cat_cutoff))

    if cat_unknown is not None:
        if not isinstance(cat_unknown, (float, str)):
            raise TypeError("The 'cat_unknown' paramater must be a number or string, depending on the metric used in transform.")

    if user_splits is not None:
        if not isinstance(user_splits, (np.ndarray, list)):
            raise TypeError("The 'user_splits' paramater must be a list or numpy.ndarray.")

    if user_splits_fixed is not None:
        if user_splits is None:
            raise ValueError("The 'user_splits' paramater must be provided.")
        else:
            if not isinstance(user_splits_fixed, (np.ndarray, list)):
                raise TypeError("The 'user_splits_fixed' paramater must be a list or numpy.ndarray.")
            elif not all(isinstance(s, bool) for s in user_splits_fixed):
                raise ValueError("The 'user_splits_fixed' paramater must be list of boolean.")
            elif len(user_splits) != len(user_splits_fixed):
                raise ValueError("Inconsistent length of 'user_splits' and 'user_splits_fixed' paramaters: {} != {}. Lengths must be equal".format(len(user_splits), len(user_splits_fixed)))

    if special_codes is not None:
        if not isinstance(special_codes, (np.ndarray, list, dict)):
            raise TypeError("The 'special_codes' paramater must be a dict, list or numpy.ndarray.")

        if isinstance(special_codes, dict) and not len(special_codes):
            raise ValueError("The 'special_codes' paramater is empty. The 'special_codes' dict must contain at least one special.")

    if split_digits is not None:
        if (not isinstance(split_digits, int) or not 0 <= split_digits <= 8):
            raise ValueError("The 'split_digits' paramater must be an integer in [0, 8]; got {}".format(split_digits))

    if mip_solver not in ("bop", "cbc"):
        raise ValueError("""The 'mip_solver' paramater must be in ("bop", "cbc"); got {}""".format(mip_solver))

    if not isinstance(time_limit, int) or time_limit < 0:
        raise ValueError("The 'time_limit' paramater must be a positive value in seconds; got {}".format(time_limit))

    if not isinstance(verbose, bool):
        raise TypeError("The 'verbose' paramater must be a boolean; got {}.".format(verbose))

    # =============================================================================
    # WOE transformation   
    # =============================================================================

    final_data = {}
    bin_table = {}
    x_woe = pd.DataFrame()
    status = pd.DataFrame()
    optimizer = ('mip', 'ls')
    for col in x_data.columns:
        temp = WoETransform(
                x = x_data[col]
                , y = y_data  
                , var_name = col
                , metric = metric
                , var_type = df_sign.loc[col, 'dtype']
                , solver = 'cp'
                , mon_constraint  = df_sign.loc[col, 'Expec']
                , divergence = divergence
                , prebinning_method = prebinning_method
                , max_n_prebins = max_n_prebins
                , min_prebin_size = min_prebin_size
                , min_n_bins = min_n_bins
                , max_n_bins = max_n_bins
                , min_bin_size = min_bin_size
                , max_bin_size = max_bin_size
                , min_bin_n_nonevent = min_bin_n_nonevent
                , max_bin_n_nonevent = max_bin_n_nonevent
                , min_bin_n_event = min_bin_n_event
                , max_bin_n_event = max_bin_n_event
                , min_event_rate_diff = min_event_rate_diff
                , max_pvalue = max_pvalue
                , max_pvalue_policy = max_pvalue_policy
                , gamma = gamma
                , outlier_detector = outlier_detector
                , outlier_params = outlier_params
                , class_weight = class_weight
                , cat_cutoff = cat_cutoff
                , cat_unknown = cat_unknown
                , user_splits = user_splits
                , user_splits_fixed = user_splits_fixed
                , special_codes = special_codes
                , split_digits = split_digits
                , mip_solver = mip_solver
                , time_limit = time_limit
                , verbose = verbose
            )
 
        k = 0
        while temp['status'] != 'OPTIMAL' and k <= 1:

            temp = WoETransform(
                            x = x_data[col]
                          , y = y_data  
                          , var_name = col
                          , metric = metric
                          , var_type = df_sign.loc[col, 'dtype']
                          , solver = optimizer[k]
                          , mon_constraint  = df_sign.loc[col, 'Expec']
                          , divergence = divergence
                          , prebinning_method = prebinning_method
                          , max_n_prebins = max_n_prebins
                          , min_prebin_size = min_prebin_size
                          , min_n_bins = min_n_bins
                          , max_n_bins = max_n_bins
                          , min_bin_size = min_bin_size
                          , max_bin_size = max_bin_size
                          , min_bin_n_nonevent = min_bin_n_nonevent
                          , max_bin_n_nonevent = max_bin_n_nonevent
                          , min_bin_n_event = min_bin_n_event
                          , max_bin_n_event = max_bin_n_event
                          , min_event_rate_diff = min_event_rate_diff
                          , max_pvalue = max_pvalue
                          , max_pvalue_policy = max_pvalue_policy
                          , gamma = gamma
                          , outlier_detector = outlier_detector
                          , outlier_params = outlier_params
                          , class_weight = class_weight
                          , cat_cutoff = cat_cutoff
                          , cat_unknown = cat_unknown
                          , user_splits = user_splits
                          , user_splits_fixed = user_splits_fixed
                          , special_codes = special_codes
                          , split_digits = split_digits
                          , mip_solver = mip_solver
                          , time_limit = time_limit
                          , verbose = verbose
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
        
    # status['name'] = status['name'].apply(lambda x: metric+'_' + x)
        
    final_data['status'] = status
    final_data['x_woe'] = x_woe
    final_data['bining_tables'] = bin_table
        
    return final_data
        
        
        
        
        
        