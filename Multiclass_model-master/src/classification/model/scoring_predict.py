"""
Module to predict the classification and provide scoring
"""

import logging
import os
from datetime import datetime
import pickle
from sklearn.metrics import roc_auc_score
from . import preprocessing, utils
import shutil


def main(cfg):
    """
    Calculating score for the fitted classification pipeline.
    In particular store all intermediate results in the  main folder 'result' defined by the cfg class.
    The main  scoring dataframe is stored in pickle format based on the format defined below in the function.

    :param      cfg: class
              custom configuration class

    """

    cfg = utils.update_config(cfg)

    if os.path.exists(cfg.scoring_dir):
        shutil.rmtree(cfg.scoring_dir)

    os.makedirs(cfg.scoring_dir)

    x, y, cfg = preprocessing.main(cfg, test=True)

    f_name = os.path.join(cfg.model_dir, str(cfg.score_name) + '_mdl.pkl')
    pipeline = pickle.load(open(f_name, 'rb'))

    logging.info('Get predictions')

    predictions = pipeline.predict_proba(x)

    # calculate AUC for testing
    if y:
        auc = roc_auc_score(y, predictions[:, 1])
        logging.info('Test AUC: {:.4f}'.format(auc))

    # Converting datetime object to string
    date_time_obj = datetime.now()
    time_stamp_str = date_time_obj.strftime("%Y%m%d_%H%M")

    logging.info('Create final csv')

    # Build up df for writing scores back
    seg_scr_model = x[['full_dt', 'contract_id']].rename({'contract_id': 'seg_scr_level_id'},
                                                         axis=1)
    seg_scr_model['seg_scr_probability'] = predictions[:, 1].round(decimals=4)
    seg_scr_model['seg_scr_model'] = cfg.score_name
    seg_scr_model['seg_scr_level_cd'] = 'contract_id'
    seg_scr_model = seg_scr_model[
        ['full_dt', 'seg_scr_model', 'seg_scr_level_cd', 'seg_scr_level_id', 'seg_scr_probability']]

    f_name = os.path.join(cfg.scoring_dir, "seg_scr_model_" + str(time_stamp_str) + ".csv")
    seg_scr_model.to_csv(f_name, encoding="utf-8", header=False, sep="|", index=False)

    logging.info('predictions correctly written')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
