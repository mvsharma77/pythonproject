import logging
import os
import argparse
#import install_module.py

sys.path.append('/projects/cc/kdqm927/PythonNotebooks/model/')
from model import train, scoring_predict, utils
from visualization import main_viz
from exp_data_analysis import main_eda
from data_extraction import extract_data
from main_config import _Config

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


def run_pipeline(cfg_path, data_path=None, model_path=None, user=None, train_flg=False, score_flg=False):
    """
    Function that applys the different steps of the classification pipeline defined in the config.yaml

    :param      cfg_path:  str
            path to the config.yaml ex: "C:User\KID\Generic_model\configs\config.yml"

    :param      data_path: str
            path to the main data dir. ex: "C:User\KID\Generic_model"

    :param      user: str
            KID for snowflake call

    :return: store data in results folder
    """

    config = _Config(cfg_path, data_path=data_path, model_path=model_path, user=user,
                     train_flg=train_flg, score_flg=score_flg)

    if config.force_proxy:
        os.environ["HTTPS_PROXY"] = "http://165.225.72.36:9400"
        os.environ["HTTP_PROXY"] = "http://165.225.72.36:9400"

    if config.sql_extract_train:
        log.info('get train dataset extraction from snowflake')
        extract_data.main(config, train=True)

    if config.sql_extract_scoring:
        log.info('get scoring dataset extraction from snowflake')
        extract_data.main(config, train=False)

    if config.exploration:
        log.info('Apply data analysis')
        main_eda.main(config)

    if config.train:
        log.info('Apply train')
        train.main(config)

    if config.visual:
        log.info('Apply visualization')
        main_viz.main(config)

    if config.scoring:
        log.info('Apply scoring')
        scoring_predict.main(config)

    utils.store_data(config)


def main():
    """
    Main functions to run the classification pipeline. It receives three arguments:
       -cf path to the config.yaml file
       -dp path to the main dir where results and data will be stored
       -user KID for snowflake access
    This function is going to call run_pipeline()
    """

    parser = argparse.ArgumentParser(description='Execute Classification from the command line')

    parser.add_argument('-cf', '--config-file', required=False, dest='config_path',
                        help='Path to config file. Must match config path')

    parser.add_argument('-dp', '--data-path', required=False, dest='data_path',
                        help='Path to data folder')

    parser.add_argument('-mp', '--model-path', required=False, dest='model_path',
                        help='Path to model')

    parser.add_argument('-full_train', '--full-train-option', required=False, dest='full_train',
                        action='store_true',
                        help='option to overwrite configuration to apply a full train pipeline')

    parser.add_argument('-full_score', '--full-score-option', required=False, dest='full_score',
                        action='store_true',
                        help='option to overwrite configuration to apply a full scoring pipeline')

    parser.add_argument('-user', '--user-KID', required=False, dest='user',
                        help='user KID')

    cmd_line_arg = parser.parse_args()

    if cmd_line_arg.config_path:
        config_path = cmd_line_arg.config_path
    else:
        config_path = os.path.join(path, 'classification/config_examples/sample_config.yaml')

    run_pipeline(config_path, data_path=cmd_line_arg.data_path, model_path=cmd_line_arg.model_path,
                 user=cmd_line_arg.user, train_flg=cmd_line_arg.full_train, score_flg=cmd_line_arg.full_score)


if __name__ == '__main__':
    main()
