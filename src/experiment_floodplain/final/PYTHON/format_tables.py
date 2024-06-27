"""Functions to format tables."""

import numpy as np
import pandas as pd
from string import Formatter

kwargs = {}

# class to format tables with dictionaries
class UnseenFormatter(Formatter):
    def get_value(self, key, args, kwds):
        if isinstance(key, str):
            try:
                return kwds[key]
            except KeyError:
                return key
        else:
            return Formatter.get_value(key, args, kwds)


def get(data, index1, index2=None):
    """Get values from specified index position of
    Pandas.DataFrame `data`, as list, excluding NaNs.
    """
    if isinstance(data.index, pd.MultiIndex):
        if isinstance(index2, str) or isinstance(index2, float):
            val = data.loc[(index1, index2)].dropna().values.tolist()
        elif isinstance(index2, list):
            val = data.loc[index1].loc[index2].dropna(axis=1).values.flatten().tolist()
    else:
        val = data.loc[index1].values.tolist()

    return val


def update(key, n_keys, values, kwargs=kwargs):
    """Create dictionary from `key`, `n_keys` and `values`
    and update whatever `kwargs `dictionary is in the global space.

    """
    keys = [f"{key}{i}" for i in range(1, n_keys + 1)]
    small_dict = dict(zip(keys, values))
    kwargs.update(small_dict)

    return kwargs


def split_dataset(df, cols, p_val=False):
    """Extract dataset of coefficients and of standard deviations
    from Pandas.DataFrame `df` and column nams `cols`. 
    Add stars to coefficients according to pvalues. 
    
    """
    df_coef = df.loc[:, (slice(None), "coef")]
    df_sd = df.loc[:, (slice(None), "std")]
    df_pval = df.loc[:, (slice(None), "P>|z|")]

    if p_val:
        # add stars to coefficient, according to p-value
        df_coef = df_coef.astype(str)
        df_coef[cols] = np.where(
            (df_pval <= 0.1) & (df_pval > 0.05), 
            df_coef + "$^{*}$", 
            df_coef)
        df_coef[cols] = np.where(
            (df_pval <= 0.05) & (df_pval > 0.01), 
            df_coef + "$^{**}$", 
            df_coef)
        df_coef[cols] = np.where((df_pval <= 0.01), df_coef + "$^{***}$", df_coef)

    return df_coef, df_sd, df_pval