"""
Module to train the pipeline:

1. Features will be divided in categorical and numerical base on the column type

2. The train input data set will be divided in train, test

3. the sklearn pipeline is created giving as input: categorical and num feaures,
    parameters of the model

4. cross validation is applied on training appling k-fold cross validation if necessary the function to implement a
    predefined split can be used

5. model is retrained with the best parameters

6. Final auc for the test set is calculated.

"""

import logging
import os

from . import utils, create_pip, grid_search
from sklearn.metrics import roc_auc_score


def get_features_list(pipeline, num_feat, cat_feat):
    """
    Get information on features:
    Provide a dictionary with all information on feature pre processing and encoding.
    creating: list of numerical features,list of categorical features,
    list of encoded variables for categorical features

    :param      pipeline: sklearn.pipeline.Pipeline
              The pipeline to cross validate

    :param      num_feat: index list
              numerical features list

    :param      cat_feat:  index list
              categorical features list

    :return:     features(dict)
              A dictionary with numerical, categorical, and all classes encoded

    """
    features = dict()
    features['numerical'] = num_feat.values
    features['categorical'] = cat_feat.values
    ct_list = pipeline.named_steps['preproc'].named_steps['columntransformer'] \
        .named_transformers_['cat_feat'].named_steps['onehotencoder'].categories_
    features['Abs_features'] = list(num_feat)

    for n_classes, classes_list in enumerate(ct_list):
        for classes in classes_list:
            features['Abs_features'].append(str(cat_feat[n_classes]) + '§§' + str(classes))
    return features


def get_best_model(pipeline, best_parameters, train, w_train):
    """
    Retrain pipeline with best parameters

    :param pipeline: sklearn pipeline
    :param best_parameters: dict
        optimized set of parameters given from cv
    :param train: list [pd.DataFrame, pd.Series]
           train_df, train_target
    :param w_train: np.array
            weights array

    :return: fitted pipeline with best parameters
    """
    pipeline.set_params(**best_parameters)
    pipeline.fit(X=train[0], y=train[1], xgb__sample_weight=w_train)
    return pipeline


def main(x, y, cfg):
    """Fitting pipeline throw k-fold cross validation given pre-processed training data
        and storing all intermediate result in folder path specified by cfg class


    :param      x: pd.DataFrame
              Training DataFrame

    :param      y: pd.Series
              Targets

    :param      cfg: class (from config.py)
              custom configurations class
    """

    if not os.path.exists(cfg.model_dir):
        os.makedirs(cfg.model_dir)

    num_feat, cat_feat = utils.get_feature_type(x[cfg.features])

    logging.info('Number of features used: %d', len(num_feat) + len(cat_feat))
    logging.info('Full list of features used in the model: %s %s', num_feat, cat_feat)
    utils.get_numerical_stats(x, num_feat, cfg)

    pipeline = create_pip.create_pipeline(num_feat, cat_feat, cfg)

    if cfg.random_split:
        train, test = utils.random_train_test_split(x=x, y=y)
    else:
        train, test = utils.custom_train_test_split(x=x, y=y, cfg=cfg)

    w_train = utils.calculate_weights(train[1], cfg.ratio_weight)

    if hasattr(cfg, 'k_folds'):
        k_folds = cfg.k_folds
    else:
        k_folds = 5

    # we need to retrain grid search since the option refit it is not working with PredefinedSplit
    grid_search_df, best_parameters = grid_search.cross_validate(pipeline=pipeline, train=train, w_train=w_train,
                                                                 params=cfg.param, k_folds=k_folds)

    final_pipeline = get_best_model(pipeline, best_parameters, train, w_train)

    features = get_features_list(final_pipeline, num_feat, cat_feat)

    # calculate final AUC for testing
    auc = roc_auc_score(test[1], final_pipeline.predict_proba(test[0])[:, 1])
    logging.info('Test AUC: {:.4f}'.format(auc))


    try:
        # store data
        utils.write_pickle(best_parameters, os.path.join(cfg.model_dir, 'best_parameters.pkl'))

        utils.write_pickle(grid_search_df, os.path.join(cfg.model_dir, 'grid_search_results.pkl'))

        logging.info('Pipeline stored in %s', cfg.model_dir)
        utils.write_pickle(final_pipeline,
                           os.path.join(cfg.model_dir, cfg.score_name + '_mdl.pkl'))

        logging.info('Feature list stored in %s', os.path.join(cfg.model_dir, 'features.pkl'))
        utils.write_pickle(features, os.path.join(cfg.model_dir, 'features.pkl'))

        logging.info('Test set stored in %s', os.path.join(cfg.model_dir, cfg.score_name +
                                                           '_test.pkl'))
        utils.write_pickle(test, os.path.join(cfg.data_dir, cfg.score_name + '_test.pkl'))

    except KeyError:
        raise KeyError('Not all files have been correctly saved')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
