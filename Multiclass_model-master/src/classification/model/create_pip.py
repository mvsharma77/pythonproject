"""
create the pipeline for the Xgboost model.
In this pipeline 2 steps are applied: pre processing and xgb model.
In the pre processing step there is a pipeline given by:
1. Column selector: select just the features contained in the config
2. column transformer that apply different steps to categorical and numerical features.

Numerical Features are imputed with a single value given in the configurations.
Categorical Features are first imputed with the string 'NaN' and than one hot encoding is applied.
"""

from sklearn import compose, impute, preprocessing
from sklearn.pipeline import make_pipeline, Pipeline
import xgboost as xgb
from . import transformers

def create_pipeline(num_feat, cat_feat, cfg):
    """
    Create and return the model classification pipeline with encoding and imputation of feature and model

    :param      num_feat: list
              numerical features name list

    :param      cat_feat:  list
              categorical features name list

    :param      cfg: class
              custom configuration class

    :return:  sklearn.pipeline.Pipeline
            model pipeline
    """

    cat_pipeline = make_pipeline(
        impute.SimpleImputer(strategy='constant', fill_value='NaN'),
        preprocessing.OneHotEncoder(categories='auto', handle_unknown='ignore')
    )

    pre_process_pipeline = make_pipeline(
        transformers.ColumnSelector(columns=cfg.features),
        compose.ColumnTransformer(transformers=[
            ('num_feat', impute.SimpleImputer(strategy='constant', fill_value=cfg.num_imputer), num_feat),
            ('cat_feat', cat_pipeline, cat_feat),
        ]),
    )

    pipeline = Pipeline(steps=[
        ('preproc', pre_process_pipeline),
        ('xgb', xgb.XGBClassifier(objective='binary:logistic'))
    ])

    return pipeline
