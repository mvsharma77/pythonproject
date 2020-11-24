"""
configuration class that assign class attributes from the yml file
"""

import os
import yaml
import ruamel.yaml as ruamel
from .model import utils
path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

class _Config:
    """class with configuration attributes given from config_file.yaml file path to data directories
        are set as attributes

    :param     cfg_file: str
            path to the configuration yml file

    :param     data_path: str
             path to the data storage

    :param    model_path: str
            path to the model directory

    :param     user: str
            KID of user for Snowflake

    :param     train_flg: str
            apply full train

    :param     score_flg: str
             apply full score

        Attributes
        ----------
        root_path_of_cgf: str
            path to th config.yaml path

        features: str
            list of feature to be used for the training

        data_dir: str
            path to the directory containing the data

        var_def: str
            path te file with the variable definitions

        main_dir: str
            path to the main directory where to store all results

        tmp_result_dir: str
            path to the directory where to store temporal, previous results

        vis_dir: str
            path to the directory where to store visualization plots

        eda_dir: str
            path to the directory where to store the exploration data analysis plots

        sql_extract_train: bool
            Flag to apply the sql extraction of train DF

        sql_extract_scoring: bool
            Flag to apply the sql extraction of test DF

        exploration: bool
            Flag to apply the exploration data analysis of the input DF

        train: bool
            Flag to apply the tarining pipeline

        visual: bool
            Flag to apply the visualization results of training DF adn model

        scoring: bool
            Flag to apply the scoring on the test DF

        """


    def __init__(self, cfg_file=path, data_path=None, model_path=None, user=None, train_flg=False, score_flg=False):
        score = ''
        self.root_path_of_cgf = None
        self.c = None
        self.data_dir = None
        self.data_input = None
        self.var_def = None
        self.main_dir = None
        self.tmp_result_dir = None
        self.vis_dir = None
        self.eda_dir = None

        try:
            with open(cfg_file) as yaml_config_file:
                attrib_list = yaml.load(yaml_config_file, Loader=yaml.FullLoader)
        except KeyError:
            raise KeyError('Configuration not correctly read')

        # get path of the configs
        for name, values in attrib_list.items():
            setattr(self, name, values)
            if (name == 'user') and (user is not None):
                self.user = user
            if (name == 'main_dir') and (data_path is not None):
                self.main_dir = data_path
            if name == 'score_name':
                score = values

        if train_flg:
            self.sql_extract_train = True  # get train dataset from snowflake
            self.sql_extract_scoring = False  # get scoring dataset from snowflake
            self.train = True  # apply the train
            self.scoring = False  # apply scoring if the model have been already trained
            self.visual = True
            self.exploration = True

        if score_flg:
            self.sql_extract_train = False  # get train dataset from snowflake
            self.sql_extract_scoring = True  # get scoring dataset from snowflake
            self.train = False  # apply the train
            self.scoring = True  # apply scoring if the model have been already trained
            self.visual = False
            self.exploration = False

        self.get_paths(self.main_dir, model_path, score, yaml_config_file)

    @staticmethod
    def get_config():
        """
        Print out the configurations used to run the model
        :return: print
        """

        config_p= os.path.join(path, 'classification','configs','config_file.yaml')
        with open(config_p) as yaml_config_file:
            doc = ruamel.load(yaml_config_file, Loader=ruamel.RoundTripLoader)
        return print(ruamel.dump(doc, Dumper=ruamel.RoundTripDumper))


    def get_paths(self, main_dir, model_path, score_name, yaml_config_file):
        """
        produce path to all main directories

        :param      main_dir: str
                path to the main directory where the data/files will be written

        :param      model_path: str
                path to the trained model if specified

        :param      score_name: str
                name of the classification model

        :param      yaml_config_file: str
                path to the directory where all configurations are
        """

        self.root_path_of_cgf = os.path.dirname(yaml_config_file.name)
        self.features = utils.get_abt_vars(os.path.join(self.root_path_of_cgf, self.abt_vars))
        self.data_input = os.path.join(main_dir, 'data', 'input_file')
        self.data_dir = os.path.join(main_dir, 'data', score_name)

        if os.path.isfile(os.path.join(self.root_path_of_cgf, 'var_def.csv')):
            self.var_def = os.path.join(self.root_path_of_cgf, 'var_def.csv')

        else:
            self.var_def = os.path.join(path, 'classification','configs','var_def.csv')

        self.main_dir = main_dir
        self.tmp_result_dir = os.path.join(main_dir, 'results_tmp', score_name)
        self.vis_dir = os.path.join(self.tmp_result_dir, 'vis_results')
        self.eda_dir = os.path.join(self.tmp_result_dir, 'eda_results')
        self.scoring_dir = os.path.join(self.tmp_result_dir, 'scoring')

        if model_path:
            self.model_dir = os.path.join(model_path)
        else:
            self.model_dir = os.path.join(self.tmp_result_dir, 'model')
