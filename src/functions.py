from __future__ import division

import json
import yaml
import joblib
from datetime import datetime

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline


class CorrelationThreshhold(BaseEstimator, TransformerMixin):
    """Removes highly correlated features."""

    def __init__(self, threshhold=0.90):
        self.threshhold = threshhold

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        corr_pairs = X.corr().reset_index().melt(id_vars="index")
        corr_pairs.columns = ["feature_1", "feature_2", "r"]

        highly_corr = corr_pairs.loc[
            (corr_pairs.r < 1) & (corr_pairs.r >= self.threshhold)
        ]
        highly_corr_sorted = sorted(
            [
                sorted([x, y]) + [z]
                for x, y, z in zip(
                    highly_corr.feature_1, highly_corr.feature_2, highly_corr.r
                )
            ]
        )
        highly_corr_sorted_df = pd.DataFrame(
            highly_corr_sorted, columns=corr_pairs.columns
        )

        highly_corr_sorted_df.drop_duplicates(inplace=True)
        self.correlated_pairs = highly_corr_sorted_df
        self.dropped_features = list(set(highly_corr_sorted_df.feature_1))

        X = X.drop(self.dropped_features, axis=1).copy()
        self.feature_names = list(X.columns)
        return X


def split_dataset(df):
    """Splits main P186 dataset into two parts:
    1. 2A3A
    2. 3A Only

    Datasets are saved in the data/processed folder.
    """
    new_columns = ["Subject", "Group"] + list(df.columns)[2:]
    df.columns = new_columns
    df = df.dropna()

    three_a = [x for x in df.columns if "2A" not in x]

    twoa_threea = df.copy()
    threea_only = df[three_a].copy()
    # create filenames for saving
    now = datetime.now().strftime("%Y%m%d")
    twoa_threea_filename = "../data/processed/{}_p186_2A3A.csv".format(now)
    threea_only_filename = "../data/processed/{}_p186_3A_Only.csv".format(now)
    # save datasets
    twoa_threea.to_csv(twoa_threea_filename, index=False)
    threea_only.to_csv(threea_only_filename, index=False)
    return twoa_threea_filename, threea_only_filename

