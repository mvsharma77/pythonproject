"""
module to get just selected features
"""

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class ColumnSelector(BaseEstimator, TransformerMixin):
    """
    Transformer to get columns selected
    """
    def __init__(self, columns):
        self.columns = columns

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        """
        transform df x selecting the features listed in self.columns

        :param x: pd.DataFrame

        :return:        filtered df
        """
        assert isinstance(x, pd.DataFrame)

        try:
            return x[self.columns]
        except KeyError:
            cols_error = list(set(self.columns) - set(x.columns))
            raise KeyError('The DataFrame does not include the columns: {}'.format(cols_error))
