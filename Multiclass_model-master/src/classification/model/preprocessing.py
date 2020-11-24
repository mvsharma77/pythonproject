"""
module to divide train and test data set and create x,y
"""
import logging
import os
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc


def select_best_feat(X, y, features, random_state=42):
    """
    Calculate best set of features that are not correlated.
    Based on the result of a simple model we choose an optimized set of features to train on.

    :param      X: pd.DataFrame
                input full dataset

    :param      y: pd.Series
                target variable series

    :param      features: list
                full list of columns name that needs to be optimized

    :param      random_state: int

    :return:     best_feat: list
              optimized list of features
    """
    model = DT()
    best_auc = 0.49
    best_feat = features[0]

    for feat in features:
        X_feat = X[feat].fillna(-99999).values.reshape(-1, 1)
        y_feat = y.values
        X_train, X_test, y_train, y_test = train_test_split(X_feat, y_feat, test_size=0.40,
                                                            random_state=random_state)

        model.fit(X_train, y_train)
        y_pred = model.predict_proba(X_test)[:, 1]

        fpr, tpr, thr = roc_curve(y_test, y_pred)
        auc_score = auc(fpr, tpr)

        if auc_score > best_auc:
            best_auc = auc_score
            best_feat = feat

    return best_feat


def select_un_corr_features(X, y, features, corr_bound=0.8):
    """
    function apply feature selection and exclude the most correlated features

    :param      X: pd.DataFrame
    :param      y: pd.Series
    :param      features: list
    :param      corr_bound:
    :return::
    """
    num_feats = X[features].select_dtypes(include=np.number).columns
    un_corr_feats = list(X[features].select_dtypes(include=object).columns)

    corr_df = X[num_feats].corr()
    selected_cols = []

    for feat in corr_df.columns:

        if feat not in selected_cols:
            corr_feats = list(corr_df[np.abs(corr_df[feat]) >= corr_bound].index)

            if len(corr_feats) > 1:
                best = select_best_feat(X, y, corr_feats)
                un_corr_feats.append(best)
                logging.info('The following set of features are highly correlated: %s', corr_feats)
                logging.info('Among them, we will only use this feature in our model: %s', best)
            else:
                un_corr_feats.append(feat)

            selected_cols += corr_feats

    return un_corr_feats


def main(cfg, test=True):
    """ Apply pre-processing on output of the extraction file.
        Based on the variable "test" the dataset is going to read the train or the test output of the data extraction.
        In particular it divides input data set from the target and impute all NaN's to np.nan.
        If required, the feature list optimization step it is going to be applied

        :param      cfg: class
                  custom configuration class

        :param      test: bool
                  apply the data preparation to the test set is " test" = True else the train

        :return:     x: pd.DataFrame
                    full preprocessed train/test dataset

        :return:     y:  pd.Series
                  Series containing the target variable data.
                  If the target is not contained in the data frame (like for testing) it is going to be set as None.

        :return:    cfg: class
                  custom configuration class with the new set of features
        """

    if test:
        try:
            df = pd.read_pickle(os.path.join(cfg.data_dir, cfg.score_name + '_score_extract.pkl'))

        except Exception:
            raise KeyError("Unable to read scoring file")
    else:
        try:
            df = pd.read_pickle(os.path.join(cfg.data_dir, cfg.score_name + '_train_extract.pkl'))

        except Exception:
            raise KeyError("Unable to read train file")

        tmp = []
        for col in cfg.features:
            if len(df[col].unique()) == 1:
                logging.warning('feature: %s has just one unique value and will get drop', col)
                df.drop(col, inplace=True, axis=1)
            else:
                tmp.append(col)

        cfg.features[:] = tmp

    # create X and Y for model fitting
    if cfg.target not in df.columns:
        y = None
    else:
        y = df[cfg.target].copy()

    x = df.fillna(np.nan).copy()

    if not test and cfg.prediction['feature_sel']:
        cfg.features = select_un_corr_features(x, y, cfg.features, cfg.prediction['corr_bound'])

    return x, y, cfg


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
