"""
getting test for run_pipeline.py
"""

import os
import shutil
import unittest
from glob import glob
from ..main_config import _Config
from ..model import fit_pipeline, preprocessing, utils
from ..visualization import main_viz


path = os.path.dirname(__file__)


class UtilsTestCase(unittest.TestCase):

    def test_update_config(self):
        cfg = _Config(os.path.join(path, 'test_config.yaml'))
        self.assertTrue(cfg.data_dir, os.path.join(cfg.main_dir, 'data', cfg.score_name))

    @classmethod
    def setUpClass(self):
        self.cfg = _Config(os.path.join(path, 'test_config.yaml'))
        self.cfg.data_dir = os.path.join(path, 'data', self.cfg.score_name)
        self.cfg.tmp_result_dir = os.path.join(path, 'results_tmp', self.cfg.score_name)
        self.cfg.model_dir = os.path.join(path, 'results_tmp', self.cfg.score_name, 'model')
        self.cfg = utils.update_config(self.cfg)
        self.cfg.vis_dir = os.path.join(path, 'results_tmp', self.cfg.score_name, 'vis_results')
        os.makedirs(self.cfg.model_dir)

    def test_pipeline_fit(self):
        x, y, self.cfg = preprocessing.main(self.cfg, test=False)
        fit_pipeline.main(x=x, y=y, cfg=self.cfg)
        self.assertIsInstance(self.cfg.features, list)
        self.assertTrue(len(self.cfg.features) > 0)
        self.assertTrue(os.path.exists(self.cfg.data_dir))
        self.assertTrue(os.path.exists(self.cfg.tmp_result_dir))
        self.assertTrue(os.path.isfile(os.path.join(self.cfg.tmp_result_dir, "model",self.cfg.score_name +
                                                    '_mdl.pkl')))
        self.assertTrue(os.path.isfile(os.path.join(self.cfg.tmp_result_dir, "model", 'features.pkl')))
        self.assertTrue(os.path.isfile(os.path.join(self.cfg.tmp_result_dir, "model",
                                                    'best_parameters.pkl')))
        self.assertTrue(os.path.isfile(os.path.join(self.cfg.tmp_result_dir, "model",
                                                    'numerical_statistics.pkl')))

    def test_viz_model(self):
        if not os.path.exists(self.cfg.vis_dir):
            os.makedirs(self.cfg.vis_dir)
        x, y, self.cfg = preprocessing.main(self.cfg, test=False)

        fit_pipeline.main(x=x, y=y, cfg=self.cfg)
        main_viz.main(self.cfg)
        self.assertTrue(os.path.isfile(os.path.join(self.cfg.vis_dir,'AUC_ROC.svg')))

        self.assertTrue(bool(glob(os.path.join(self.cfg.vis_dir, 'Confusion_Matrix_all')+'*')))
        self.assertTrue(os.path.isfile(os.path.join(self.cfg.vis_dir,
                                                    'Prediction_distribution.svg')))

    @classmethod
    def tearDownClass(self):
        shutil.rmtree(os.path.join(path, 'results_tmp'))


if __name__ == '__main__':
    unittest.main()
