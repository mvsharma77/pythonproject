"""
Module to apply gird search on the sklearn pipeline during the fit
"""
import logging
import pandas as pd
from. import utils

from sklearn.model_selection import GridSearchCV



def cross_validate(pipeline, train, w_train, params,  k_folds = 5, e_val= None, scoring = 'score'):
    """Apply cross validation grid search for parameter tuning.
    This function permits to apply k-fold cross validation if the parameter e_val = None
    or to apply grid search in a fix set of train eval in case in the future the classification will
    be time dependent and therefore the sets: trin and eval need to be fixed (Predefine Split).
    In addition the parameters given in config.yml will contain the list of parameters that we want to loop over.

    :param      pipeline: sklearn.pipeline.Pipeline
              The pipeline to cross validate

    :param      train: [pd.DataFrame, pd.DataFrame]
              list with Training data, training target

    :param      e_val: [pd.DataFrame,pd.DataFrame]
              list with Evaluation data, Evaluation target

    :param      w_train: np.array
              array with weights for Training

    :param      params: dict
              parameters for Grid Search

    :param      scoring: str
              type of scoring we would like to aply our pipeline optimization


    :return    gs_total_df: pd.df
                full information about the grid search

    :return:    best_params: best parameters list out of the grid search

    """

    if e_val:
        cv_x, cv_y, pred_split = utils.create_predefined_split(train, e_val)
        cv = pred_split
    else:
        cv_x = train[0]
        cv_y = train[1]
        cv = k_folds

    parameters = {}
    for parm in params:
        if isinstance(params[parm], list):
            parameters['xgb__' + parm] = params[parm]
        else:
            parameters['xgb__' + parm] = [params[parm]]

    grid_search = GridSearchCV(estimator=pipeline,
                               param_grid=parameters,
                               refit=True,
                               scoring='roc_auc',
                               cv=cv,
                               return_train_score=True,
                               verbose=2,
                               n_jobs=1)

    grid_search.fit(X=cv_x, y=cv_y, xgb__sample_weight=w_train)

    cv_results = pd.DataFrame(grid_search.cv_results_)
    gs_df = cv_results[['mean_train_score', 'std_train_score', 'mean_test_score', 'std_test_score',
                        'mean_fit_time', 'mean_score_time', 'params']]

    gs_param = pd.DataFrame(gs_df['params'].values.tolist(), index=gs_df.index)
    gs_total_df = pd.concat([cv_results[['mean_train_score', 'mean_test_score', 'mean_fit_time',
                                         'mean_score_time']], gs_param], axis=1) \
        .rename(columns={'mean_train_score': 'train_roc_auc', 'mean_test_score': 'eval_roc_auc',
                         'mean_fit_time': 'train_time', 'mean_score_time': 'inference_time'})

    best_results = cv_results.iloc[grid_search.best_index_]
    score_names = ['mean_train_', 'std_train_', 'mean_test_', 'std_test_']
    score_names = [score_name + scoring for score_name in score_names]
    result_string = 'Best model: \ntraining-AUC: {:.4f} +- {:.4f}\n eval-AUC: {:.4f} +- {:.4f}'
    result_string += '\nBest params: {}'
    logging.info(result_string.format(*best_results[score_names], grid_search.best_params_))

    return gs_total_df, grid_search.best_params_
