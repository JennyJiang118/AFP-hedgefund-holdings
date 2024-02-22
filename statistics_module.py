import statsmodels.api as sm
import pandas as pd

def ols(df, y, xs, intercept=True, dropna=True):
    """
    Perform Ordinary Least Squares regression using statsmodels.

    Args:
        df: DataFrame containing the data.
        y: The dependent variable.
        xs: List of independent variables.
        intercept: Boolean, if True, adds an intercept to the model.
        dropna: Policy for handling NaNs. True corresponds to missing='drop'
    
    Returns:
        Fitted OLS model.
    """
    X = df[xs]
    y = df[y]
    if intercept:
        X = sm.add_constant(X)

    if dropna:
        model = sm.OLS(y, X, missing=True)
    else:
        model = sm.OLS(y, X)

    results = model.fit()
    return results


def wls(df, y, xs, weights, intercept=True, dropna=True):
    """
    Perform Weighted Least Squares regression using statsmodels.

    Args:
        df: DataFrame containing the data.
        y: The dependent variable.
        xs: List of independent variables.
        intercept: Boolean, if True, adds an intercept to the model.
        dropna: Policy for handling NaNs. True corresponds to missing='drop'
        weights: Weights for WLS.

    Returns:
        Fitted WLS model.
    """
    X = df[xs]
    y = df[y]
    if weights is not None:
        weights = df[weights]
    if intercept:
        X = sm.add_constant(X)

    if dropna:
        missing_policy = 'drop'
    else:
        missing_policy = 'none'

    if weights is not None:
        model = sm.WLS(y, X, weights=weights, missing=missing_policy)
    else:
        model = sm.OLS(y, X, missing=missing_policy)
        
    results = model.fit()
    return results
