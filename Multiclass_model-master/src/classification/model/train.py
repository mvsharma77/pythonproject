"""
module to train the Xgboost model:
 1. The default configurations are updatet with the parameters contained in the config.yam
 2. The train set is split in main dataset (x_var) and target (y_var)
 3. the pipeline that is defined in create_pip.py is fitted
"""

import logging


from . import fit_pipeline, preprocessing, utils


def main(cfg):
    """
    Training the classification model:
    Takes a fixed set of training input files (defined in the extraction) and trains a classification model pipeline

    :param      cfg: class
              custom configuration class

    """

    cfg = utils.update_config(cfg)

    x_var, y_var, cfg = preprocessing.main(cfg, test=False)
    fit_pipeline.main(x=x_var, y=y_var, cfg=cfg)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
