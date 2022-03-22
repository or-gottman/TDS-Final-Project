#data processing
import pandas as pd

#Patterns Mining
from efficient_apriori import apriori
from sklearn import preprocessing


def label_encoding(df, ordinal_columns):
    label_encoded_df = df.copy(deep=True)
    for column in ordinal_columns:
        # label_encoder object knows how to understand word labels
        label_encoder = preprocessing.LabelEncoder()
        # Encode labels in column
        label_encoded_df[column] = label_encoder.fit_transform(label_encoded_df[column])
    return label_encoded_df


def one_hot_encoding(df, categorical_columns):
    other_columns = set(df.columns) - set(categorical_columns)
    onehot_df = df[other_columns]
    # Convert all categorical columns to one hot encoding
    for column in categorical_columns:
        onehot_col = pd.get_dummies(df[column], prefix=column)
        onehot_df = pd.concat([onehot_df, onehot_col], axis=1)
    return onehot_df

def preprocess_df(df):

    dtf = df.copy()

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



def mine_labeled_df_association_rules(dtf, label_title):

    df = dtf.copy()

    # split dataframe to partitions, based on the label
    partitions_dict = dict(iter(df.groupby(label_title)))

    # we now have a dictionary (partitions_dict), where the keys are the different labels,
    # and their values are the corresponding dataframe rows that have that label

    # converting to transactions:
    records, transactions = dict(), dict()
    for key in partitions_dict.keys():
        records[key] = partitions_dict[key].to_dict(orient='records')
        transactions[key] = []
        for r in records[key]:
            transactions[key].append(list(r.items()))

    # mine association rules, using apriori algorithm, from every df independently, and add to list of total rules
    total_rules = list()
    for key in transactions.keys():
        itemsets, rules = apriori(transactions[key], min_support=0.5, min_confidence=0.8)
        total_rules.extend(rules)

    return total_rules


def mine_association_rules(dtf):
    records = dtf.to_dict(orient='records')
    transactions = []
    for r in records:
        transactions.append(list(r.items()))

    itemsets, rules = apriori(transactions, min_support=0.5, min_confidence=0.8)
    return rules
