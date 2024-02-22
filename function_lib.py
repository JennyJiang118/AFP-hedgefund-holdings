from collections import namedtuple
import pickle
import pandas as pd
import scipy
import constants as c
import matplotlib.pyplot as plt
import numpy as np
import statistics_module
import statsmodels.api as sm
from IPython.display import display
import seaborn as sns
from scipy.cluster import hierarchy
from scipy import stats
from contextlib import contextmanager
import os
from pandas.io.formats.style import Styler
from scipy.optimize import minimize
import mgarch
from scipy.stats import kurtosis, skew, jarque_bera
from typing import List

FAKE_REG_RES = namedtuple('FAKE_REG_RES', ['params', 'rsquared', 'tvalues'])

def build_corr_heatmap(signal_df, others, factors, indicator=None, method='spearman'):
    # factors: list, eg. ['signal']. several / single interested factors.
    # others: list, risk factors & other factors.
    if indicator:
        df = signal_df[signal_df['Indicator'] == indicator]
    else:
        df = signal_df.copy()
    corr_df = df.groupby(c.DATE)[list(factors) + list(others)].corr(method=method)
    corr_df = corr_df.unstack().mean().unstack()[factors]
    if len(factors) == 1:
        # only one interested factors
        hmap = corr_df.mul(100).round(1)[1:]
    else:
        hmap = corr_df.mul(100).round(1)
    return hmap


def get_z_scores(df, xs, by=None, method='daily', winsorize=True, ln=False, normal=False, suffix=''):
    """ Transform signals to z-scores
    Returns:
        df: dataframe containing signals after transformation
    """
    winsorize = _get_winsorize_threshold(winsorize)
    for x in xs:
        if method == 'daily': 
            if by in df.columns:
                group_results = df.groupby([c.DATE, by])[x].apply(_z_function, winsorize, ln, normal)
            else:
                group_results = df.groupby(c.DATE)[x].apply(_z_function, winsorize, ln, normal)
        elif method == 'panel': 
            if by in df.columns:
                group_results = df.groupby(by)[x].apply(_z_function, winsorize, ln, normal)
            else:
                group_results = _z_function(df[x], winsorize, ln, normal)
        else:
            raise ValueError(f"Unrecognized method: {method}.")
        
        # Add new: Reset the index of the results to match the DataFrame's index
        group_results = group_results.reset_index(level=by, drop=True) if by in df.columns else group_results
        group_results.index = df.index  
        df[x + suffix] = group_results

    return df


def orthogonalize_signals(frame, ys, xs, lny=False, lnx=False, suffix='', by=c.DATE):
    """get signals orthogonal to other factors"""
    df = frame.copy()
    if by not in df.columns:
        raise KeyError(f"by col {by} is not in the dataframe.")
    for y in ys:
        if lny is True:
            df[y] = (np.log(np.abs(df[y]))).replace(-np.inf, np.nan, inplace=True)
        if lnx is True: 
            for x in xs:
                df[x] = (np.log(np.abs(df[x]))).replace(-np.inf, np.nan, inplace=True)
        frame[y + suffix] = df.groupby(by).apply(_get_reg_residuals, y, xs).reset_index(level=by, drop=True)
    return frame


def run_backtest(df, signal_col, lags=[0, 1, 2, 3, 4, 5], return_col=['AlphaReturn', 'ExMarketReturn'], WLS_model=True):
    """
    Run backtest based on fitted daily cross-sectional factor models

    Attention to: signal_col, return_col, whether to do neutralization & neutralize_cols.
    
    Returns:
        df: dataframe, backtest results objects by lags and Return type (AlphaReturn / ExMarketReturn)
    """
    dataframes = {}
    folder_name = f'{signal_col}_backtest'.replace('/', 'by')

    reg_df = df.copy()
    reg_df.sort_values(by=c.DATE, inplace=True)
    dir_path = os.path.join(folder_name, 'Total Data')
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Data handling: e.g. start date > 2017, etc.
    reg_df.sort_values(by=[c.DATE], inplace=True)
    for lag in lags:
        results = pd.DataFrame()
        params = pd.DataFrame()
        rsq = pd.DataFrame()
        for ycol in return_col:
            # groupby by c.SEC_ID to lag X
            if check_df_has_columns(reg_df, ['InvSpecRisk']) and WLS_model:
                cs_reg_model = build_factor_models(df=reg_df, return_col=ycol, signal_cols=signal_col,
                                            groupby_cols=[c.SEC_ID], winsorize_return=False,
                                            weight='InvSpecRisk', shift=lag + 1)
                # print('using WLS')
            else:
                cs_reg_model = build_factor_models(df=reg_df, return_col=ycol, signal_cols=signal_col,
                                            groupby_cols=[c.SEC_ID], winsorize_return=False,
                                            weight=None, shift=lag + 1)
                # print('using OLS')
            result_table, return_params, params_rsq = evaluate_factor_models(cs_reg_model)
            results = pd.concat([results, result_table], ignore_index=True)
            params = pd.concat([params, return_params], axis=1)
            params_rsq.name = ycol + 'rsq'
            rsq = pd.concat([rsq, params_rsq], ignore_index=True)
        results.index = return_col
        results.index.name = 'ReturnType'
        # rsq = pd.DataFrame(rsq.values.reshape(-1, 2), columns=return_col)  # if single factor regression
        dataframes[lag] = results

    # save for plot usage
    with open(os.path.join(dir_path, 'rsq.pkl'), 'wb') as f:
        pickle.dump(rsq, f)
    with open(os.path.join(dir_path, 'params.pkl'), 'wb') as f:
        pickle.dump(params, f)

    df = pd.concat(dataframes, names=['lag', 'ReturnType'])
    df.index.names = ['lag', 'ReturnType']
    df = df.swaplevel('lag', 'ReturnType')
    df = df.sort_index()
    df.to_pickle(os.path.join(dir_path, 'backtest_table.pickle'))
    other_cols = list(set(df.columns) - set(['Mean Nobs']))
    display(Table(df, title=f'{signal_col} backtest').format_numbers('{:.2f}', subset=other_cols).
            format_numbers('{:.0f}', subset='Mean Nobs').set_widths('unset'))

    avg_exposure_corr = factor_turnover(reg_df, dir_path=dir_path, exposure_col=signal_col)

    return df


def factor_turnover(signal_df, dir_path, indicator=None, exposure_col='revision', period=10):
    # Compute exposure correlation decay
    if indicator:
        df = signal_df[(signal_df['Indicator'] == indicator)]
    else:
        df = signal_df.copy()
    exp = df.drop_duplicates(subset=[c.DATE, c.SEC_ID], keep='last').pivot(index=c.DATE, columns=c.SEC_ID, values=exposure_col)

    with Timer('correlation decay'):
        exp_corr = pd.DataFrame({i: exp.shift(i).T.corrwith(exp.T) for i in range(period+1)})
        # print(exp_corr.mean().loc[1:period])
        # print(exp_corr)
        A, K = fit_exp_nonlinear(exp_corr.mean().loc[1:period])
    
    ax = exp_corr.mean().plot(label='Average Exposure Autocorrelation')
    t = exp_corr.columns
    ax.plot(t, model_func(t, A, K), label=f'Exponential fit through L={period}')
    ax.grid(True)
    ax.legend()
    ax.set_title(f'Exposure Correlation Decay ({exposure_col})')
    ax.set_ylabel('Correlation')
    ax.set_xlabel('Lag')
    plt.show()

    return exp_corr.mean()


def build_factor_models(df, return_col, signal_cols, groupby_cols=None, winsorize_return=True, weight=None, 
                        shift=None, shift_y=False, rolling_option=False, by=None):
    """
    Fit daily cross-sectional factor models

    models: series, regression results objects by date
    """
    if groupby_cols is None:
        groupby_cols = [c.SEC_ID]
        # Period could be ['FiscalPeriod', 'FiscalPeriodYear', 'FiscalPeriodMonth']
    signal_cols = maybe_to_list(signal_cols)
    winsorize_return = _get_winsorize_threshold(winsorize_return)

    cols = [c.DATE, return_col, *signal_cols]
    if weight:
        cols.append(weight)
    if shift:
        cols = cols + list(groupby_cols)
    check_df_has_columns(df, cols)

    reg_df = df[cols].copy()
    if by:
        bycol = [c.DATE, by]
    else:
        bycol = c.DATE
    if shift is not None:
        if shift_y:
            reg_df[return_col] = reg_df.groupby(groupby_cols)[return_col].shift(-shift)
            if rolling_option:
                reg_df['rolling_forward_y'] = reg_df.groupby(groupby_cols)[return_col].transform(
                    lambda x: x.rolling(window=shift, min_periods=1).mean())
                reg_df = reg_df.dropna()
                models = reg_df.groupby(bycol).apply(
                    lambda x: _cross_sectional_reg(x, 'rolling_forward_y', signal_cols, winsorize_return, weight))
            else:
                reg_df = reg_df.dropna()
                models = reg_df.groupby(bycol).apply(
                    lambda x: _cross_sectional_reg(x, return_col, signal_cols, winsorize_return, weight))
        else:
            for signal in signal_cols:
                reg_df[signal] = reg_df.groupby(groupby_cols)[signal].shift(shift)
            reg_df = reg_df.dropna()
            models = reg_df.groupby(bycol).apply(
                lambda x: _cross_sectional_reg(x, return_col, signal_cols, winsorize_return, weight))
    else:
        reg_df = reg_df.dropna()
        models = reg_df.groupby(bycol).apply(
            lambda x: _cross_sectional_reg(x, return_col, signal_cols, winsorize_return, weight))
    return models


def evaluate_factor_models(models, signal_names=None):
    """
    Derive the summary statistics for factor performance from the fitted models.
    """
    # Extract parameters and statistics from the models
    valid_models = models.loc[~models.apply(lambda x: x.rsquared).isna()]
    params = valid_models.apply(lambda x: x.params)
    rsq = valid_models.apply(lambda x: x.rsquared)
    tvalues = valid_models.apply(lambda x: x.tvalues)
    nobs = valid_models.apply(lambda x: x.nobs)
    
    if signal_names is None:
        signal_names = [i for i in params.columns if i != 'const']
    else:
        signal_names = signal_names if isinstance(signal_names, list) else [signal_names]    
    params = params.loc[:, signal_names]
    tvalues = tvalues.loc[:, signal_names]
    
    # Calculate summary statistics
    annual_return = PerformanceMetrics.annualized_return(params)
    annual_volatility = PerformanceMetrics.annualized_volatility(params)
    sharpe = PerformanceMetrics.sharpe_ratio(params)
    max_drawdown = PerformanceMetrics.max_drawdown(params)
    t_stat = annual_return / annual_volatility
    abs_tvalues_greater_2 = (tvalues.abs() > 2).sum() / tvalues.count()
    
    # Compile results into a DataFrame
    results = pd.DataFrame({
        'Ann Ret%': annual_return * 100,
        'Ann Vol%': annual_volatility * 100,
        'Sharpe': sharpe,
        'Max DD%': max_drawdown * 100,
        'T Stat': t_stat,
        '#|T|>2': abs_tvalues_greater_2 * 100,
        'Mean R^2%': rsq.mean() * 100,
        'Max R^2%': rsq.max() * 100,
        'Mean Nobs': int(nobs.mean())
    })
    
    return results, params, rsq


def fit_exp_nonlinear(series):
    t, y = series.index, series.values
    opt_params, param_cov = scipy.optimize.curve_fit(model_func, t, y, maxfev=1000)
    A, K = opt_params
    return A, K


def model_func(t, A, K):
    return A * np.exp(-t / K)


def check_df_has_columns(df, columns):
    """
    Check if the dataframe contains all the specified columns.
    
    Args:
    df (pd.DataFrame): The DataFrame to check.
    columns (list of str): The list of columns to check for in the DataFrame.
    
    Raises:
    ValueError: If any specified column is not found in the DataFrame.
    """
    missing_columns = [col for col in columns if col not in df.columns]
    if missing_columns:
        print(f'Missing columns {missing_columns}')
        # raise ValueError(f"The following columns are missing from the DataFrame: {missing_columns}")
        return False
    else:
        return True


def _cross_sectional_reg(frame, y, xs, winsorize_y, weight, min_obs=10):
    x = maybe_to_list(xs)
    frame[y] = frame[y].clip(frame[y].quantile(winsorize_y[0]), frame[y].quantile(winsorize_y[1]))
    if len(frame.dropna()) < max(min_obs, len(xs) + 1):
        empty_sr = pd.Series(np.full(len(xs) + 1, np.nan), index=xs + ['const'])
        return FAKE_REG_RES(empty_sr, np.nan, empty_sr)
    return statistics_module.wls(frame, y, xs, weight, dropna=True)


def _get_reg_residuals(df, y, xs):
    mdl = statistics_module.ols(df, y, xs, intercept=True, dropna=True)
    predictions = mdl.predict(sm.add_constant(df[mdl.model.exog_names[1:]]))
    residuals = df[y] - predictions
    return residuals


def _z_function(x: pd.Series, winsorize, ln, normal):
    if normal:
        return x.rank(pct=True).apply(lambda a: stats.norm.ppf(a)).clip(-3, 3)
    else:
        x = x.clip(x.quantile(winsorize[0]), x.quantile(winsorize[1]))
        x = (np.log(np.abs(x))).replace(-np.inf, np.nan) if ln else x
    return ((x - x.dropna().mean()) / x.dropna().std()).clip(-3, 3)


def _get_winsorize_threshold(winsorize_arg):
    # TODO: add splitting alpha returns and removing extremes by idiosyncratic risk
    if winsorize_arg is True:
        winsorize_arg = (0.01, 0.99)
    elif winsorize_arg is False:
        winsorize_arg = (0, 1)
    else:
        assert isinstance(winsorize_arg, (tuple, list)) and len(winsorize_arg) == 2, f'must be 2-tuple: {winsorize_arg}'
    return winsorize_arg


#########################################  Performance Metrics  ######################################### 
class PerformanceMetrics:
    @staticmethod
    def annualized_return(factor_returns, periods_per_year=None):
        """
        Calculate the annualized return of a returns series.
        
        Args:
            factor_returns: pd.Series or pd.DataFrame, daily returns series to calculate annualized return.
            periods_per_year: int, number of periods per year for annualization (default is 252 for trading days).
        
        Returns:
            float or pd.Series: the annualized return.
        """
        if not periods_per_year:
            if c.FREQUENCY == 'D':
                periods_per_year = 252
            elif c.FREQUENCY == 'M':
                periods_per_year = 12
            elif c.FREQUENCY == 'Q':
                periods_per_year = 4
            else:
                raise ValueError('Trading frequency set error')
        mean_daily_return = factor_returns.mean()
        return mean_daily_return * periods_per_year

    @staticmethod
    def annualized_volatility(factor_returns, periods_per_year=None):
        """
        Calculate the annualized volatility of a returns series.
        
        Args:
            factor_returns: pd.Series or pd.DataFrame, daily returns series to calculate annualized volatility.
            periods_per_year: int, number of periods per year for annualization (default is 252 for trading days).
        
        Returns:
            float or pd.Series: the annualized volatility.
        """
        if not periods_per_year:
            if c.FREQUENCY == 'D':
                periods_per_year = 252
            elif c.FREQUENCY == 'M':
                periods_per_year = 12
            elif c.FREQUENCY == 'Q':
                periods_per_year = 4
            else:
                raise ValueError('Trading frequency set error')
        daily_volatility = factor_returns.std(ddof=1)
        return daily_volatility * np.sqrt(periods_per_year)

    @staticmethod
    def sharpe_ratio(factor_returns, risk_free_rate=0.0, periods_per_year=None):
        """
        Calculate the Sharpe ratio of a returns series.
        
        Args:
            factor_returns: pd.Series or pd.DataFrame, daily returns series for Sharpe ratio calculation.
            risk_free_rate: float, the risk-free rate per period (default is 0.0).
            periods_per_year: int, number of periods per year for annualization (default is 252 for trading days).
        
        Returns:
            float or pd.Series: the Sharpe ratio.
        """
        if not periods_per_year:
            if c.FREQUENCY == 'D':
                periods_per_year = 252
            elif c.FREQUENCY == 'M':
                periods_per_year = 12
            elif c.FREQUENCY == 'Q':
                periods_per_year = 4
            else:
                raise ValueError('Trading frequency set error')
        excess_returns = factor_returns - risk_free_rate / periods_per_year
        sharpe_ratio = excess_returns.mean() / excess_returns.std(ddof=1)
        return sharpe_ratio * np.sqrt(periods_per_year)

    @staticmethod
    def max_drawdown(factor_returns):
        """
        Calculate the maximum drawdown of a returns series.
        
        Args:
            factor_returns: pd.Series or pd.DataFrame, daily returns series to calculate the maximum drawdown.
        
        Returns:
            float or pd.Series: the maximum drawdown.
        """
        cumulative_returns = (1 + factor_returns).cumprod()
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns - running_max) / running_max
        return drawdown.min()

    @staticmethod
    def win_rate(factor_returns):
        """
        Calculate the win rate (percentage of positive returns).
        
        Args:
            factor_returns: pd.Series or pd.DataFrame, daily returns series to calculate the win rate.
        
        Returns:
            float: the win rate.
        """
        return (factor_returns > 0).sum() / len(factor_returns)


#########################################  Utils  ######################################### 
def maybe_to_list(list_var):
    return list_var if isinstance(list_var, list) else [list_var]


def df_to_parquet(df, file_name):
    df.to_parquet(f'{file_name}.parquet.gzip', compression='gzip')


def read_cache(file_name):
    return pd.read_parquet(f'{file_name}.parquet.gzip')


def save_plot(ax, dir_path, file_name):
    fig = ax.get_figure()
    os.makedirs(dir_path, exist_ok=True)
    file_path = os.path.join(dir_path, file_name)
    fig.savefig(file_path)


class CustomStyler(Styler):
    def format_numbers(self, formatter, subset=None):
        return self.format(formatter, subset=subset)
    
    def set_widths(self, width, subset=None):
        if width == 'unset':
            width = 'auto'
        if subset is not None:
            styles = [{'selector': f'th:contains("{col}"), td:nth-child({i+1})', 
                       'props': [('width', width)]} for i, col in enumerate(subset)]
        else:
            styles = [{'selector': 'th, td', 'props': [('width', width)]}]
        return self.set_table_styles(styles, overwrite=False)


def Table(df, title, column_width='100px', number_fmt="{:.2f}", subset=None):
    styled_df = CustomStyler(df)
    styled_df = styled_df.background_gradient(axis=None)
    if title:
        styled_df = styled_df.set_caption(title)
    if number_fmt:
        formatter = {col: number_fmt for col in (subset if subset is not None else df.columns)}
        styled_df = styled_df.format(formatter)
    styled_df = styled_df.set_widths(column_width)
    return styled_df


import time

class Timer:
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, type, value, traceback):
        self.end = time.time()
        self.interval = self.end - self.start
        if self.name:
            print(f"{self.name} RunTime: {self.interval:.4f} seconds")
        else:
            print(f"Elapsed time: {self.interval:.4f} seconds")
