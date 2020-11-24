"""
getting test for classification/model/
"""

import unittest
import os
import pandas as pd
import numpy as np
import shutil
from ..model import utils
from ..main_config import _Config

path = os.path.dirname(__file__)


class UtilsTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        create class with updated configuration from test.yml
        """
        cls.cfg = _Config(os.path.join(path, 'test_config.yaml'))
        cls.cfg = utils.update_config(cls.cfg)
        cls.cfg.data_dir = os.path.join(path, 'data', cls.cfg.score_name)
        cls.cfg.model_dir = os.path.join(path, 'results_tmp', cls.cfg.score_name, 'model')
        if not os.path.exists(cls.cfg.model_dir):
            os.makedirs(cls.cfg.model_dir)

    def test_get_abt_vars(self):
        list_v = list(self.cfg.features)
        self.assertEqual(len(list_v), 18)
        self.assertTrue(all(isinstance(i, str) for i in list_v))

    def test_calculate_weights(self):
        s1 = pd.Series([1, 0, 1], dtype='float')
        self.assertCountEqual(utils.calculate_weights(s1, 0.15), np.array([0.15, 1.00, 0.15]))

    def test_get_numerical_stats(self):
        x = pd.DataFrame({'num_col': [1, 1, 1, 2, 3, 6]})

        temp_file_path = os.path.join(self.cfg.model_dir, 'numerical_statistics.pkl')

        try:
            utils.get_numerical_stats(x, ['num_col'], self.cfg)
            contents = pd.read_pickle(temp_file_path)
        finally:
            files = os.listdir(self.cfg.model_dir)
            if not files:
                os.rmdir(self.cfg.model_dir)
            os.remove(temp_file_path)

        self.assertCountEqual(contents.columns, ['std', 'mean', 'imputer'])
        self.assertEqual(contents['std'].values[0], np.std([1, 1, 1, 2, 3, 6], ddof=1))
        self.assertEqual(contents['mean'].values[0], np.mean([1, 1, 1, 2, 3, 6]))

    def test_get_feature_type(self):
        x = pd.DataFrame({'num_col': [1, 1, 1, 2, 3, 6, np.nan], 'cat_col': [1, 2, 'a', np.nan,
                                                                             'n', 'c', 'p']})
        self.assertCountEqual(utils.get_feature_type(x), ['num_col', 'cat_col'])

    @classmethod
    def tearDownClass(self):
        shutil.rmtree(os.path.join(path, 'results_tmp'))

