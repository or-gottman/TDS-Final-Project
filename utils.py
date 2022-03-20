#data processing
import pandas as pd
import numpy as np
import scipy as sp

def preprocess_df(dtf):

    # Defining numeric and categorical columns
    numeric_columns = dtf.dtypes[(dtf.dtypes == "float64") | (dtf.dtypes == "int64")].index.tolist()
    very_numerical = [nc for nc in numeric_columns if dtf[nc].nunique() > 20]
    categorical_columns = [c for c in dtf.columns if c not in numeric_columns]
    ordinals = list(set(numeric_columns) - set(very_numerical))

    # Filling Null Values with the column's mean
    na_columns = dtf[very_numerical].isna().sum()
    na_columns = na_columns[na_columns > 0]
    for nc in na_columns.index:
        dtf[nc].fillna(dtf[nc].mean(), inplace=True)

    # Dropping and filling NA values for categorical columns:
    # 1. Drop if at least 70% are NA:
    nul_cols = dtf[categorical_columns].isna().sum() / len(dtf)
    drop_us = nul_cols[nul_cols > 0.7]

    # 2. Fill with a new 'na' category:
    dtf = dtf.drop(drop_us.index, axis=1)
    categorical_columns = list(set(categorical_columns) - set(drop_us.index))

    return dtf

