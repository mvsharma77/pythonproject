import logging
import os
import pickle
import shutil

from .Viz import Viz


def main(config):

    viz_dir = config.vis_dir

    with open(os.path.join(config.model_dir, config.score_name + '_mdl.pkl'), 'rb') as file:
        pipeline = pickle.load(file)
    preprocessor = pipeline.named_steps['preproc']
    model = pipeline.named_steps['xgb']

    with open(os.path.join(config.data_dir, config.score_name + '_test.pkl'), 'rb') as file:
        data, label = pickle.load(file)
        label = label.eq('Yes').mul(1)
    with open(os.path.join(config.model_dir, 'features.pkl'), 'rb') as file:
        feats_info = pickle.load(file)

    num_feat = feats_info['numerical']
    cat_feat = feats_info['categorical']
    abs_feat = feats_info['Abs_features']

    with open(os.path.join(config.model_dir, 'numerical_statistics.pkl'), 'rb') as file:
        num_feat_stats = pickle.load(file)

    if os.path.exists(viz_dir):
        shutil.rmtree(viz_dir)

    os.makedirs(viz_dir)

    viz_obj = Viz(preprocessor, model, data, label, num_feat, cat_feat, abs_feat, num_feat_stats,
                  viz_dir, config.var_def)

    if config.vis['imp']['t_flg']:
        viz_obj.save_feat_imp_table(**config.vis['imp']['t_param'])

    if config.vis['imp']['p_flg']:
        viz_obj.save_feat_imp_plot(**config.vis['imp']['p_param'])

    if config.vis['auc']['flg']:
        viz_obj.plot_auc_roc(**config.vis['auc']['param'])

    if config.vis['em']['flg']:
        viz_obj.plot_eval_metrics(**config.vis['em']['param'])

    if config.vis['cf_mat']['flg']:
        viz_obj.plot_conf_matrix(**config.vis['cf_mat']['param'])

    if config.vis['shap']['flg']:
        viz_obj.plot_shapely_values(**config.vis['shap']['param'])

    if config.vis['decile']['flg']:
        viz_obj.bin_score_deciles(**config.vis['decile']['param'])

    if config.vis['dist']['flg']:
        viz_obj.plot_pred_distribution(**config.vis['dist']['param'])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
