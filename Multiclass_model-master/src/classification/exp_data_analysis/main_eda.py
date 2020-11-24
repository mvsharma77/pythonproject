import logging
import os
import pickle
import shutil
from .eda import Eda


def main(config):

    with open(os.path.join(config.data_dir, config.score_name + '_train_extract.pkl'), 'rb') as file:
        data = pickle.load(file)

    if os.path.exists(config.eda_dir):
        shutil.rmtree(config.eda_dir)

    os.makedirs(config.eda_dir)

    eda_obj = Eda(data, config.features, config.target, config.eda['classes'],
                  config.eda_dir, config.var_def)

    if config.eda['num']['flg']:
        eda_obj.plot_numerical(**config.eda['num']['param'])

    if config.eda['cat']['flg']:
        eda_obj.plot_categorical(**config.eda['cat']['param'])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
