"""
module with utils functions used in the other modules
"""
import logging
import os
import pickle
import numpy as np
import pandas as pd
import shutil
from datetime import datetime as dt

import glob
from sklearn.model_selection import train_test_split, PredefinedSplit


def random_train_test_split(x, y, test_size=0.2):
    """
    Get random split between train, test

    :param      x: pd.DataFrame
            Training Data Frame

    :param      y: pd.Series
            Targets

    :param      cfg: config class

    :param      test_size: float
            ratio between data and test

    :return:     [train_x, train_y]: list(pd.DataFrame,pd.Series)

    :return:     [test_x, test_y]: list(pd.DataFrame,pd.Series)
        """

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=1)
    return [x_train, y_train], [x_test, y_test]


def custom_train_test_split(x, y, cfg):
    """
    Get data test eval split based on the DATE/ TIME reported in config file
    split data: <= 31.12.2017 for training (and evaluating), >= 31.01.2018 for testing

    :param    x: pd.DataFrame
            Training Data Frame

    :param    y: pd.Series or list-like
            Targets

    :param    cfg: config class

    :param    eval_size:  float
            ratio between data and test

    :return:     [train_x, train_y]: list(pd.DataFrame,pd.Series)

    :return:     [test_x, test_y]: list(pd.DataFrame,pd.Series)
     """

    # test data
    is_test = (x['full_dt'] >= pd.to_datetime(cfg.last_date_test).date())
    # data and eval data
    is_train = ~is_test
    # data and eval
    x_train = x.loc[is_train, :]
    y_train = y[is_train]
    # test
    x_test = x.loc[is_test, :]
    y_test = y[is_test]
    logging.info('Test shape with %i rows', x_test.shape[0])

    return [x_train, y_train], [x_test, y_test]


def random_train_test_eval_split(x, y, test_size=0.2, eval_size=0.2):
    """Get random split between train, test, eval set
    The train and eval are set.

    :param     x: pd.DataFrame
            Training Data Frame

    :param     y: pd.Series or list-like
            Targets

    :param    cfg: config class

    :param    test_size: float            ratio between data and test

    :param    eval_size:  float
            ratio between train and eval


    :return:     [train_x, train_y]: list(pd.DataFrame,pd.Series)

    :return:     [eval_x, eval_y]: list(pd.DataFrame,pd.Series)

    :return:     [test_x, test_y]: list(pd.DataFrame,pd.Series)

     """

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=1)

    x_train, x_eval, y_train, y_eval = train_test_split(x_train, y_train, test_size=eval_size, random_state=1)

    return [x_train, y_train], [x_eval, y_eval], [x_test, y_test]


def train_test_eval_split(x, y, cfg, eval_size=0.25):
    """

    Get data test eval split based on the date reported in config file
    split data: <= 31.12.2017 for training (and evaluating), >= 31.01.2018 for testing

    :param      x: pd.DataFrame
            Training Data Frame

    :param      y: pd.Series or list-like
            Targets

    :param      cfg: config class

    :param      eval_size:  float
            ratio between data and test

    :return:     [train_x, train_y]: list(pd.DataFrame,pd.Series)

    :return:     [eval_x, eval_y]: list(pd.DataFrame,pd.Series)

    :return:     [test_x, test_y]: list(pd.DataFrame,pd.Series)

    """

    # test data
    is_test = (x['full_dt'] >= pd.to_datetime(cfg.last_date_test).date())
    # data and eval data
    is_train_eval = ~is_test
    # data and eval
    x_train, x_eval, y_train, y_eval = train_test_split(x.loc[is_train_eval, :],
                                                        y[is_train_eval],
                                                        test_size=eval_size,
                                                        random_state=1)
    # test
    x_test = x.loc[is_test, :]
    y_test = y[is_test]
    logging.info('Test shape with %i rows', x_test.shape[0])

    return [x_train, y_train], [x_eval, y_eval], [x_test, y_test]


def get_feature_type(x):
    """
    Get list of numerical and categorical pandas columns type to generate numerical and categorical features

    :param x: pd.DataFrame
            Training DataFrame

    :return: num_feat: list
            list of numerical columns

    :return: cat_feat: list
            list of categorical columns
    """

    num_feat = x.select_dtypes(include=np.number).columns
    cat_feat = x.select_dtypes(include=object).columns

    # bool_feat = df.select_dtypes(include='bool').columns
    assert x.shape[1] == len(num_feat) + len(cat_feat)

    logging.info('Numerical features list: %s', num_feat)
    logging.info('Categorical features list: %s', cat_feat)

    try:
        return num_feat, cat_feat
    except KeyError:
        cols_error = list(set(x.columns) - set(num_feat) - set(cat_feat))
        raise KeyError('The Data Frame does not include the columns: {}'.format(cols_error))


def create_predefined_split(train, e_val):

    """
    create a predefined split in order to use a fix train and eval set in the cross validation.

    :param      train: [pd.DataFrame, pd.DataFrame]
        list with Training data, training target

    :param      e_val: [pd.DataFrame,pd.DataFrame]
        list with Evaluation data, Evaluation target

    :return:     cv_x: pd.DataFrame
                train eval df

    :return:     cv_y: pd.DataFrame
                train, eval target

    :return:     pred_split: list
                matrix to define the set of train and eval set

    """
    cv_x = pd.concat([train[0], e_val[0]], axis=0, ignore_index=True)
    cv_y = pd.concat([train[1], e_val[1]], axis=0, ignore_index=True)

    logging.info('Train set with %d  rows.', train[0].shape[0])
    logging.info('Evaluation set with %d rows ', e_val[0].shape[0])

    # Set data eval set in the PredefinedSplit
    # The indices which have the value -1 will be kept in data.
    train_indices = np.full((len(train[0]),), -1, dtype=int)

    # The indices which have zero or positive values, will be kept in test
    eval_indices = np.full((len(e_val[0]),), 0, dtype=int)
    eval_fold = np.array(np.append(train_indices, eval_indices))
    pred_split = PredefinedSplit(test_fold=eval_fold)

    return cv_x, cv_y, pred_split


def calculate_weights(y_train, ratio_w):
    """
    Assign weights to records when label is 1

    :param     y_train: pd.Series target

    :param     ratio_w: float number

    :return: create array of weight associated to the parameters

    """

    return np.where(y_train == 1, ratio_w, 1)  # according to fraction in data set


def get_numerical_stats(x, num_feat, cfg):
    """
    Create DataFrame with std mean and imputation information for all numerical features

    :param  x: pd.Series target

    :param  num_feat: list str of numerical features

    :param   cfg: class
             configuration

    :return:   df_num_feat:  pd.df
            statistics for  numerical variables

    """

    df_num_feat = x[num_feat].agg(['std', 'mean']).T
    df_num_feat['imputer'] = cfg.num_imputer
    f_name = os.path.join(cfg.model_dir, 'numerical_statistics.pkl')
    df_num_feat.to_pickle(f_name)
    logging.info('Set Features stored in %s', f_name)
    return df_num_feat


def get_abt_vars(variables_file):
    """
    Get list of features used in the model from file "abt_vars.txt"

    :param      variables_file: str
            path to file

    :return:    vars_selected:     list
    """

    with open(variables_file, 'r') as my_file:
        vars_selected = my_file.read().replace('\n', '').split(',')
       # vars_selected = [x.lower() for x in vars_selected]

    return vars_selected


def write_pickle(i_object, path_obj):
    """
    Write object to pickle

    :param      i_object: object

    :param      path_obj: str
               path to store the result
    """

    try:
        with open(path_obj, 'wb') as handle:
            pickle.dump(i_object, handle, protocol=pickle.HIGHEST_PROTOCOL)
    except KeyError:
        raise KeyError('The object was not correctly saved in: {}'.format(path_obj))


def update_config(config):
    """
    Update class config with the variables present in yaml file

    :param   config: class
                configuration class

    :return: class defined in model/config.py updated with new features list

    """

    keys_prediction = list(config.prediction.keys())
    if not config.prediction['features']:
        keys_prediction.remove('features')

    for key in keys_prediction:
        config.__setattr__(key, config.prediction[key])

    assert config.target not in config.features, "Error! target_variable in the feature list"

    if not os.path.exists(config.tmp_result_dir):
        os.makedirs(config.tmp_result_dir)
    return config


def store_data(config):
    """
    store configurations to the result dir and all temporal data

    :param config: configuration class containing all paths

    """
    # Store additional config files
    curr_time = dt.now().strftime("%Y.%m.%d_%H.%M")
    result_dir = os.path.join(config.main_dir, 'results', config.score_name, curr_time)
    os.makedirs(result_dir)
    os.makedirs(os.path.join(result_dir, 'config'))

    # copy config
    for file in glob.glob(os.path.join(config.root_path_of_cgf, "*.txt"), recursive=True):
        shutil.copy(file, os.path.join(result_dir, 'config'))

    for file in glob.glob(os.path.join(config.root_path_of_cgf, "*.sql"), recursive=True):
        shutil.copy(file, os.path.join(result_dir, 'config'))

    for file in glob.glob(os.path.join(config.root_path_of_cgf, "*.yaml"), recursive=True):
        shutil.copy(file, os.path.join(result_dir, 'config'))

    try:
        if config.train:
            shutil.copytree(config.model_dir,  os.path.join(result_dir, 'model'))

        if config.visual:
            if not os.path.exists(os.path.join(result_dir, 'model')):
                shutil.copytree(config.model_dir, os.path.join(result_dir, 'model'))
            shutil.copytree(config.vis_dir, os.path.join(result_dir, 'vis_results'))

        if config.exploration:
            shutil.copytree(config.eda_dir, os.path.join(result_dir, 'eda_results'))

        if config.scoring:
            if not os.path.exists(os.path.join(result_dir, 'model')):
                shutil.copytree(config.model_dir, os.path.join(result_dir, 'model'))
            shutil.copytree(config.scoring_dir, os.path.join(result_dir, 'scoring'))

        logging.info('All results have been saved in %s', result_dir)

    except KeyError:
        raise KeyError('The configurations are not correctly saved in: {}'.format(result_dir))
