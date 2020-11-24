import os
import itertools

import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import xgboost as xgb
import shap

from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.metrics.ranking import _binary_clf_curve
from sklearn.pipeline import Pipeline


class Config:
    """ A Configuration class that contains some constants used during visualization"""

    colors = {'TRQ': (0.36, 0.76, 0.8), 'RED': (0.92, 0.11, 0.04),
              'YEL': (0.89, 0.88, 0), 'BOR': (0.69, 0.02, 0.05)}

    MAX_STR_LEN = 35

    TIT_FS = 25
    AXS_FS = 20
    TIC_FS = 15

    FONT = {'family': 'Arial', 'weight': 'normal', 'size': 8}


matplotlib.rc('font', **Config.FONT)


class Viz:
    """ A Result visualization class built for tree based machine learning models.
       the visualizations are based on SHAP along with some basic visualization plots.

    :param preprocessor: Pipeline
            the SK-Learn pipeline that was used to process the data during training.

    :param     model: xgb.sklearn.XGBClassifier
            the tree based model produced during training. Although the input is of class
            xgb.sklearn.XGBClassifier, we extract from it the corresponding xgb.core.Booster
            and use it instead.

    :param     data: pd.DataFrame
            A pandas DataFrame for the data used for visualization. Please keep in mind that
            some of the visualization functions only need the model, however other functions
            based on SHAP need some sample data. By data here we means only the input features
            to the model.

    :param     label: np.ndarray
            A numpy one dimensional array that contains the values of the target variable which
            correspond to each data point given by the pandas DataFrame in data.

    :param    raw_num_feats: list
            It contains a list of the names of the numerical features used by the model.

    :param     raw_cat_feats: list
            It contains a list of the names of the categorical features used by the model

    :param     abs_feats: list
            It contains a list of absolute features in the same order that enters the model.
            The order is as follows: first numerical features, then the encoded categorical
            features. The name of an encoded categorical feature is structured as follows:
            'raw_cat_feat' + '§§' + 'category'

    :param    num_feats_stats: pd.DataFrame
            A pandas DataFrame that contain some statistics about the numerical features.
            These statistics were extracted from the training data.

    :param     viz_dir: str
            A string for the directory that will be used to store the visualization plots
            and tables

    :param     var_def_file: str
            A path to a csv file that contain the English and German explanations of each
            of the features used by the model

        Attributes
        ----------
        data_enc: scipy.sparse.csr.csr_matrix or np.ndarray
            A two dimensional array that contains the stores the encoded version of the
            input data

        var_def_en: dict
            A dictionary the store the English explanation for each raw feature

        var_def_de: dict
            A dictionary the store the German explanation for each raw feature

        model_feats_map: dict
            A dictionary that maps the encoded features used by the model to their absolute
            name given in abs_feats. Please keep in mind that the XGB booster name the features
            as f1, f2, ...

        raw_feats_map: dict
            A dictionary that maps the name of each encoded features to the corresponding raw
            feature. This means that a numerical feature is mapped to itself, while a categorical
            feature of the form: 'raw_cat_feat' + '§§' + 'category' is mapped to 'raw_cat_feat'

        pred_score: np.ndarray
            A one dimensional array that contains the CHURN prediction scores (probabilities).
        """

    def __init__(self, preprocessor: Pipeline, model: xgb.sklearn.XGBClassifier, data: pd.DataFrame,
                 label: np.ndarray, raw_num_feats: list, raw_cat_feats: list, abs_feats: list,
                 num_feats_stats: pd.DataFrame, viz_dir: str, var_def_file: str = 'var_def.csv'):

        self.preprocessor = preprocessor
        self.model = model.get_booster()

        try:
            assert (len(self.model.feature_names) == len(abs_feats))
        except Exception:
            raise Exception('Model and features do not match.')

        self.data_raw = data
        self.true_label = label
        self.data_enc = self.preprocessor.transform(self.data_raw)

        self.raw_num_feats = raw_num_feats
        self.num_feats_stats = num_feats_stats
        self.raw_cat_feats = raw_cat_feats
        self.abs_feats = abs_feats

        self.var_def = pd.read_csv(var_def_file)
        self.var_def['RAW_VAR'] = self.var_def#.str.lower()
        self.var_def_en = dict(zip(self.var_def['RAW_VAR'], self.var_def['VAR_DEF_EN']))
        self.var_def_de = dict(zip(self.var_def['RAW_VAR'], self.var_def['VAR_DEF_DE']))

        self.viz_dir = viz_dir

        self.raw_feats_map = dict(zip(self.abs_feats, [feat.split('§§')[0] for feat in
                                                       self.abs_feats]))
        self.model_feats_map = dict(zip(['f' + str(i) for i in range(
            len(self.model.feature_names))], self.abs_feats))
        self.pred_score = self.model.predict(xgb.DMatrix(self.data_enc))

    @staticmethod
    def split_string_lines(input_str, max_len=Config.MAX_STR_LEN):

        """This method splits the input string into lines such that the max number of characters
        in each line is determined by max_len. It is important to keep in mind that the split of
        the input string is done based on spaces.

        :param input_str: str
            input string to be split Ex:"A Group of Data for each line"

        :param max_len: int
            the max length allowed in each line, Ex: 10

        :return It return a new string split into lines: "A Group of Data for each line" .
        """

        str_list = input_str.split()
        if len(str_list) <= 1:
            return input_str
        else:
            output_str = str_list[0]
            line_len = len(output_str)
            for _str in str_list[1:]:
                if line_len + len(_str) > max_len:
                    output_str += '\n' + _str
                    line_len = len(_str)
                else:
                    output_str += ' ' + _str
                    line_len = len(output_str.split('\n')[-1])
            return output_str

    def save_feat_imp_table(self, aggregated=False, save_format='csv', fname='feat_imp'):

        """This method saves the feature importance data frame of the model to disk.

        :param aggregated: bool
            Whether the feature importance are aggregated over raw features or left as
            absolute features

        :param save_format: str
            the format of the file to be saved either .csv or .pkl

        :param fname: str
            the name of the file used for saving it to disk

        :return None, as It saves the feature importance data frame to disk.
        """

        imp_df = self.get_feat_imp_df(aggregated=aggregated)
        imp_df.sort_values('VAR_IMP', ascending=False, inplace=True)

        if save_format == 'csv':
            imp_df.to_csv(os.path.join(self.viz_dir, fname + '.csv'))
        else:
            imp_df.to_pickle(os.path.join(self.viz_dir, fname + '.pkl'))

    def save_feat_imp_plot(self, aggregated=True, x=('RAW_VAR',), y='VAR_IMP_REL',
                           x_label='Model Features', y_label='Relative Importance',
                           fname='Feature_Importance_', str_len=(Config.MAX_STR_LEN,),
                           plot_format='.pdf'):


        """This method saves the feature importance bar plot of the model to disk.

        :param aggregated: bool
            Whether the feature importance are aggregated over raw features or left as
            absolute features

        :param x: tuple of str
            A tuple of strings that contains the name of the columns to plot the feature
            importance diagram before, it can be ('RAW_VAR', 'VAR_DEF_EN', 'VAR_DEF_DE')

        :param y: str
            A tuple of string the contains which feature importance value should be plotted
            it can be ('VAR_IMP_REL', 'VAR_IMP')

        :param x_label: str
            A string for the x_label to be put on the plot

        :param y_label: str
            A string for the x_label to be put on the plot

        :param fname: str
            the name of the file used for saving it to disk, extension is default .pdf

        :param str_len: tuple of int
            This integer define the maximum number of charcaters allowed per line so that
            very long strings are split into multiple lines

        :param plot_format: str
            This defines the format used to save the plot '.png', '.jpg', '.pdf'

        :return None, It saves the feature importance plot to disk.
        """

        for _x, _len in zip(x, str_len):
            imp_df = self.get_feat_imp_df(aggregated=aggregated, str_len=_len)
            imp_df.sort_values('VAR_IMP', ascending=True, inplace=True)
            fig, ax = plt.subplots(figsize=(len(imp_df), len(imp_df)))
            ax = imp_df.plot(x=_x, y=y, kind='barh', ax=ax, legend=False,
                             color=Config.colors['TRQ'])
            for p in ax.patches:
                width, height = p.get_width(), p.get_height()
                x_pos, y_pos = p.get_xy()
                ax.annotate('{:.2f}'.format(width), (x_pos + width + 0.01, y_pos + .2 * height),
                            color='black')

            ax.set_title('Feature Importance of the Model \n', fontsize=Config.TIT_FS,
                         fontweight='bold')
            ax.set_ylabel(x_label, fontsize=Config.AXS_FS)
            ax.set_xlabel(y_label, fontsize=Config.AXS_FS)
            ax.tick_params(axis='y', which='major', labelsize=Config.TIC_FS)

            _fname = fname + _x
            fig.savefig(os.path.join(self.viz_dir, _fname + plot_format), bbox_inches='tight')
            plt.close()

    def get_feat_imp_df(self, aggregated=False, str_len=1000):

        """This method return a DataFrame for the feature importance. The absolute and relative
        importance are added as two columns: ('VAR_IMP', 'VAR_IMP_REL'). IF the aggregated option
        is set to False, only one column is added for each feature which is the absolute name of
        the feature. If aggregated is set to True, then categorical features are aggregated into
        one feature and three columns are added to the DataFrame: ('RAW_VAR', 'VAR_DEF_EN',
        'VAR_DEF_DE). If no the English or the German definitions are not available, the RAW_VAR is
        used for both of them.

        :param aggregated: bool
            Whether the feature importance are aggregated over raw features or left as
            absolute features

        :param str_len: int
            the maximum number of characters allowed per line while plotting

        :return pd.DataFrame, It returns a pandas DataFrame that contains feature importance of the model.
        """

        imp_df = pd.DataFrame(self.model.get_fscore(), index=['VAR_IMP'])
        imp_df.rename(columns=self.model_feats_map, inplace=True)
        imp_df = imp_df.transpose().reset_index()
        imp_df.columns = ['ABS_VAR', 'VAR_IMP']
        imp_df['VAR_IMP_REL'] = imp_df['VAR_IMP'] / imp_df['VAR_IMP'].max()
        if aggregated:
            imp_df['RAW_VAR'] = imp_df['ABS_VAR'].map(self.raw_feats_map)
            agg_df = imp_df.groupby('RAW_VAR')['VAR_IMP'].sum().reset_index()
            agg_df['VAR_IMP_REL'] = agg_df['VAR_IMP'] / agg_df['VAR_IMP'].max()

            agg_df['VAR_DEF_EN'] = agg_df['RAW_VAR'].map(self.var_def_en)
            agg_df['VAR_DEF_EN'] = np.where(agg_df['VAR_DEF_EN'].isna(), agg_df['RAW_VAR'],
                                            agg_df['VAR_DEF_EN'])
            agg_df['VAR_DEF_EN'] = np.vectorize(Viz.split_string_lines)(agg_df['VAR_DEF_EN'],
                                                                        str_len)
            agg_df = agg_df[['VAR_DEF_EN'] + list(agg_df.columns)[:-1]]

            agg_df['VAR_DEF_DE'] = agg_df['RAW_VAR'].map(self.var_def_de)
            agg_df['VAR_DEF_DE'] = np.where(agg_df['VAR_DEF_DE'].isna(), agg_df['RAW_VAR'],
                                            agg_df['VAR_DEF_DE'])
            agg_df['VAR_DEF_DE'] = np.vectorize(Viz.split_string_lines)(agg_df['VAR_DEF_DE'],
                                                                        str_len)
            agg_df = agg_df[['VAR_DEF_DE'] + list(agg_df.columns)[:-1]]

            return agg_df
        else:
            imp_df.rename(columns={'ABS_VAR': 'RAW_VAR'}, inplace= True)

        return imp_df

    def plot_eval_metrics(self, plot_size=8, fname='ACC_PRE_REC_F1', table=True,
                          save_format_table='csv', metrics=('ACC', 'PRE', 'REC', 'F1'),
                          plot_format='.pdf'):

        """This method saves a plot of the requested metrics at different thresholds for the data
        used to create the object. It also save a table of the values used to create the plot if
        it is requested.

        :param plot_size: int
            Dimensions of the plot, it is always a square plot

        :param fname: str
            the name of the file used for saving it to disk

        :param table: bool
            Whether a table of the metrics should be saved beside the plot or not

        :param save_format_table: str
            the format of the file to be saved either .csv or .pkl

        :param metrics: tuple ('ACC', 'PRE', 'REC', 'F1')
            Contains the different metrics to be plotted, you can only select from those 4.

        :param plot_format: str
            This defines the format used to save the plot '.png', '.jpg', '.pdf'

        :return None, It saves a plot of the requested metrics at different thresholds, these metrics are
        calculated with respect to the data used to initiate the instance of the class viz.
        """

        fps, tps, thr = _binary_clf_curve(self.true_label, self.pred_score)
        tns, fns = fps[-1] - fps, tps[-1] - tps
        precision = tps / (tps + fps)
        recall = tps / tps[-1]
        accuracy = (tns + tps) / (fps[-1] + tps[-1])
        f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)

        if table:
            metrics_df = pd.DataFrame({'Threshold': thr, 'Accuracy': accuracy,
                                       'Precision': precision, 'Recall': recall,
                                       'F1_Score': f1_score})
            if save_format_table == 'csv':
                metrics_df.to_csv(os.path.join(self.viz_dir, fname + '.csv'))
            else:
                metrics_df.to_pickle(os.path.join(self.viz_dir, fname + '.pkl'))

        fig, ax = plt.subplots(figsize=(plot_size, plot_size))
        if 'ACC' in metrics:
            ax.plot(thr, accuracy, color=Config.colors['RED'], lw=2, label='Accuracy')
        if 'PRE' in metrics:
            ax.plot(thr, precision, color=Config.colors['TRQ'], lw=2, label='Precision')
        if 'REC' in metrics:
            ax.plot(thr, recall, color=Config.colors['YEL'], lw=2, label='Recall')
        if 'F1' in metrics:
            ax.plot(thr, f1_score, color='black', lw=2, label='F1_Score', linestyle='-')

        ax.set_title('Model Evaluation Metrics', fontsize=Config.TIT_FS, fontweight='bold')
        ax.set_xlim([0.0, thr.max() + 0.01])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Probability Threshold', fontsize=Config.AXS_FS)
        ax.set_ylabel('Evaluation Metrics Scores', fontsize=Config.AXS_FS)
        ax.legend(loc="best")

        fig.savefig(os.path.join(self.viz_dir, fname + plot_format), bbox_inches='tight')
        plt.close()

    def plot_auc_roc(self, plot_size=5, fname='AUC_ROC', table=True, save_format='csv',
                     plot_format='.png'):

        """This method saves a plot of area under the receiver operating curve at different thresholds
        for the data used to create the object. It also save a table of the values used to create the plot if
        it is requested.

        :param plot_size: int
            Dimensions of the plot, it is always a square plot

        :param fname: str
            the name of the file used for saving it to disk

        :param table: bool
            Whether a table of the metrics should be saved beside the plot or not

        :param save_format: str
            the format of the file to be saved either .csv or .pkl

        :param plot_format: str
            This defines the format used to save the plot '.png', '.jpg', '.pdf'

        :return None, It saves a plot of the requested metrics at different thresholds, these metrics are
        calculated with respect to the data used to initiate the instance of the class viz.
        """

        fpr, tpr, thr = roc_curve(self.true_label, self.pred_score)
        auc_score = auc(fpr, tpr)

        if table:
            auc_df = pd.DataFrame({'Threshold': thr, 'True_Positive_Rate': tpr,
                                   'False_Positive_Rate': fpr})
            if save_format == 'csv':
                auc_df.to_csv(os.path.join(self.viz_dir, fname + '.csv'))
            else:
                auc_df.to_pickle(os.path.join(self.viz_dir, fname + '.pkl'))

        fig, ax = plt.subplots(figsize=(plot_size, plot_size))
        ax.plot(fpr, tpr, color=Config.colors['RED'], lw=2, label='AUC (%0.3f)' % auc_score)
        ax.set_title('Model Receiver Operating Curve', fontsize=Config.TIT_FS, fontweight='bold')
        ax.plot([0, 1], [0, 1], color=Config.colors['TRQ'], lw=2, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=Config.AXS_FS)
        ax.set_ylabel('True Positive Rate', fontsize=Config.AXS_FS)

        ax.legend(loc="lower right")

        fig.savefig(os.path.join(self.viz_dir, fname + plot_format), bbox_inches='tight')
        plt.close()

    def get_conf_matrix(self, threshold=None, normalize_axis=None):

        """This method calculates the confusion matrix for the model with respect to the given
        data based on the given threshold and the normalization axis. If the threshold is None,
        then it is calculated based on Youden formula. The normalize_axis can take values of
        ('None', 'pred', 'all', 'true')

        :param threshold: float or str 'None'
            The threshold used to decide the class of the prediction scores, if 'None' then
            it is calculated using Youden formula.

        :param normalize_axis: str
            the axis on which the confusion matrix is normalized, it can be on all values or
            with respect to true label or with respect to predictions or without normalization
            at all.

        :return tuple, It returns a tuple of confusion matrix as numpy array and the threshold as a float.
        """

        if threshold == 'None':
            fpr, tpr, thr = roc_curve(self.true_label, self.pred_score)
            youden_thr = tpr + (1 - fpr) - 1
            threshold = thr[youden_thr.argmax()]

        opt_pred = np.where(self.pred_score > threshold, 1, 0)
        conf_mat = confusion_matrix(self.true_label, opt_pred)

        if normalize_axis == 'pred':
            conf_mat = conf_mat.astype('float') / conf_mat.sum(axis=0)
        elif normalize_axis == 'true':
            conf_mat = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]
        elif normalize_axis == 'all':
            conf_mat = conf_mat.astype('float') / conf_mat.sum()

        return conf_mat, threshold

    def plot_conf_matrix(self, threshold=('None',), plot_size=10, classes=('Stay', 'General_Churn'),
                         normalize_axis=('None',), cmap=plt.cm.Reds, fname='Confusion_Matrix_',
                         plot_format='.pdf'):

        """This method calculates the confusion matrix for the model with respect to the given
        data based on the given threshold and the normalization axis. If the threshold is None,
        then it is calculated based on Youden formula. The normalize_axis can take values of
        ('None', 'pred', 'all', 'true')

        :param threshold: tuple of int and str='None'
            a tuple of the different thresholds required for the confusion matrix to be plotted

        :param plot_size: int
            Dimensions of the plot, it is always a square plot

        :param classes : tuple of str
            the names of the binary classes the model is predicting

        :param normalize_axis: tuple of str
            a tuple of the normalization required for each plot

        :param cmap: matplotlib color map
            the color map used to plot the confusion matrix

        :param fname: str
            the name of the file used for saving it to disk

        :param plot_format: str
            This defines the format used to save the plot '.png', '.jpg', '.pdf'

        :return None, It saves the requested confusion matrix on disk.
        """

        for thre, axis in zip(threshold, normalize_axis):
            conf_mat, thre_conf = self.get_conf_matrix(threshold=thre, normalize_axis=axis)
            fig, ax = plt.subplots(figsize=(plot_size, plot_size))
            plt.imshow(conf_mat, interpolation='nearest', cmap=cmap)

            norm = 'None'
            if axis == 'pred':
                norm = 'On Predictions'
            elif axis == 'true':
                norm = 'On True Label'
            elif axis == 'all':
                norm = 'On Total'
            ax.set_title('Confusion Matrix at Threshold: %0.3f \n (Normalized: %s)' % (thre_conf, norm),
                         fontsize=Config.TIT_FS, fontweight='bold')


            plt.colorbar()
            tick_marks = np.arange(len(classes))
            plt.xticks(tick_marks, classes, fontsize=Config.TIC_FS)
            plt.yticks(tick_marks, classes, fontsize=Config.TIC_FS)
            plt.ylim(1.5, -0.5)
            plt.xlim(-0.5, 1.5)

            fmt = '.1%' if axis in ['true', 'pred', 'all'] else 'd'
            color_thr = conf_mat.max() / 2
            for i, j in itertools.product(range(conf_mat.shape[0]), range(conf_mat.shape[1])):
                plt.text(j, i, format(conf_mat[i, j], fmt), horizontalalignment="center",
                         color="white" if conf_mat[i, j] > color_thr else "black",
                         fontsize=Config.TIC_FS)

            ax.set_ylabel('True label', fontsize=Config.AXS_FS)
            ax.set_xlabel('Predicted label', fontsize=Config.AXS_FS)
            plt.tight_layout()

            _fname = fname + str(axis) + '_%0.03f' %thre_conf

            fig.savefig(os.path.join(self.viz_dir, _fname + plot_format), bbox_inches='tight')
            plt.close()

    def plot_shap_categorical(self, cat_feat, shap_df, pos, cut_off, cat_len, plot_type, y_label,
                              x_label, label_len, fdir, title_map, title_len, plot_format='.pdf'):

        """This method is used to plot the shapely values of a categorical feature, for a given categorical
        feature, the shaply values are calculated as the mean shaply value of each category

        :param cat_feat: str
            The name of the raw categorical feature for which the shaply values are to be plotted

        :param shap_df: pd.DataFrame
            A pandas DataFrame that contain the shaply values for each data point (rows) and each absolute
            feature (columns)

        :param pos: int
            The order of the feature in the feature importance of the model

        :param cut_off: float
            It defines the minimum mean shaply values for a category to be plotted, for example if the mean
            shapely values for category A is less than the cut_off, this category is excluded.

        :param cat_len: int
            The maximum number of characters allowed per line for the name of the categories appear on the y-axis

        :param plot_type: str
            It defines the type of the plot used to plot the shaply values, it can be either 'box' for box plotting
            or 'bar', default is 'bar'.

        :param y_label: str
            It defines what to be written on the Y-axis of the plot, either German, English or
            Raw feature name or any other given string

        :param x_label: str
            It defines what to be written on the X-axis of the plot

        :param label_len: int
            The maximum number of characters allowed per line for the y_label

        :param fdir: str
            The path to the directory where the plot is stored

        :param title_map: str
            It defines which mapping is used to name the title of the plot, default uses the raw feature name,
            you can either choose 'DE' for German mapping or 'EN' for  English mapping.

        :param title_len: int
            The maximum number of characters allowed per line for the title of the plot

        :param plot_format: str
            This defines the format used to save the plot '.png', '.jpg', '.pdf'

        :return None, It saves the requested plot on disk
        """

        features = [feat for feat in shap_df.columns if feat.startswith(cat_feat + '§§')]
        agg = shap_df[features].mean(axis=0).sort_values()
        agg = agg[agg.abs() >= cut_off]

        if len(agg) > 0 and plot_type == 'box':
            agg = shap_df[list(agg.index)].copy()
            agg.columns = [feat.split(cat_feat + '§§')[1] for feat in agg.columns]
            agg.columns = [Viz.split_string_lines(feat, cat_len) for feat in agg.columns]
            fig, ax = plt.subplots(figsize=(5, 0.8 * len(agg.columns)))
            sns.boxplot(data=agg, orient='h', color=Config.colors['TRQ'], ax=ax)

            if (title_map == 'EN') and (cat_feat in self.var_def_en):
                title = Viz.split_string_lines(self.var_def_en[cat_feat], title_len)
                title = "Shap: %s \n" % title

            elif (title_map == 'DE') and (cat_feat in self.var_def_de):
                title = Viz.split_string_lines(self.var_def_de[cat_feat], title_len)
                title = "Shap: %s \n" % title
            else:
                title = "Shap: %s \n" % cat_feat

            ax.set_title(title, fontsize=Config.TIT_FS, fontweight='bold')
            if y_label in ['DE', 'EN', 'RF']:
                if (y_label == 'EN') and (cat_feat in self.var_def_en):
                    ax.set_ylabel(Viz.split_string_lines(self.var_def_en[cat_feat], label_len),
                                  fontsize=Config.AXS_FS)
                elif (y_label == 'DE') and (cat_feat in self.var_def_de):
                    ax.set_ylabel(Viz.split_string_lines(self.var_def_de[cat_feat], label_len),
                                  fontsize=Config.AXS_FS)
                else:
                    ax.set_ylabel(cat_feat, fontsize=Config.AXS_FS)
            else:
                ax.set_ylabel(y_label, fontsize=Config.AXS_FS)

            ax.set_xlabel(x_label, fontsize=Config.AXS_FS)
            ax.tick_params(axis='y', which='major', labelsize=Config.TIC_FS)
            ax.text(1.2, -0.08, 'Min Shap: \n %0.3f' % cut_off, transform=ax.transAxes,
                    ha='center', va='center', bbox=dict(facecolor='white', alpha=0.5),
                    fontsize=10, color='black')
            fig.savefig(os.path.join(fdir, '%03d_%s_box' % (pos, cat_feat) + plot_format),
                        bbox_inches='tight')
            plt.close()

        elif len(agg) > 0:
            agg.index = agg.reset_index()["index"].str.split(cat_feat + '§§', expand=True)[1].values
            agg.index = np.vectorize(Viz.split_string_lines)(agg.reset_index()["index"], cat_len)
            fig, ax = plt.subplots(figsize=(5, 0.8 * len(agg)))
            ax = agg.plot.barh(color=Config.colors['TRQ'], ax=ax)

            if (title_map == 'EN') and (cat_feat in self.var_def_en):
                title = Viz.split_string_lines(self.var_def_en[cat_feat], title_len)
                title = "Shap: %s \n" % title
            elif (title_map == 'DE') and (cat_feat in self.var_def_de):
                title = Viz.split_string_lines(self.var_def_de[cat_feat], title_len)
                title = "Shap: %s \n" % title
            else:
                title = "Shap: %s \n" % cat_feat

            ax.set_title(title, fontsize=Config.TIT_FS, fontweight='bold')
            if y_label in ['DE', 'EN', 'RF']:
                if (y_label == 'EN') and (cat_feat in self.var_def_en):
                    ax.set_ylabel(Viz.split_string_lines(self.var_def_en[cat_feat], label_len),
                                  fontsize=Config.AXS_FS)
                elif (y_label == 'DE') and (cat_feat in self.var_def_de):
                    ax.set_ylabel(Viz.split_string_lines(self.var_def_de[cat_feat], label_len),
                                  fontsize=Config.AXS_FS)
                else:
                    ax.set_ylabel(cat_feat, fontsize=Config.AXS_FS)
            else:
                ax.set_ylabel(y_label, fontsize=Config.AXS_FS)
            ax.set_xlabel(x_label + ' (mean) ', fontsize=Config.AXS_FS)
            ax.tick_params(axis='y', which='major', labelsize=Config.TIC_FS)
            ax.text(1.2, -0.05, 'Min Shap: \n %0.3f' % cut_off, transform=ax.transAxes,
                    ha='center', va='center', fontsize=8, color='black')
            fig.savefig(os.path.join(fdir, '%03d_%s' % (pos, cat_feat) + plot_format),
                        bbox_inches='tight')
            plt.close()


    def plot_shap_numerical(self, feat, shap_df, sample, pos, interaction, opacity, mean_na,
                            percentage, y_label, x_label, label_len, fdir, title_map, cmap,
                            title_len, plot_format='.pdf'):

        """This method is used to plot a dependency plot for the shapely values of a given feature (Numerical).
        This plot only consider the data points where the feature has a value, for the data points where the
        value of the feature is NA, the mean shaply value is computed and written at the top right corner of
        the plot. Moreover, the dependency plot only consider the data points around the mean within an n std.
        The value of this n is calculated such that the selected point represent a given percentage of the
        given data points

        :param feat: str
            The name of the raw (Numerical) feature for which the dependency plot of the shaply values is plotted

        :param shap_df: pd.DataFrame
            A pandas DataFrame that contain the shaply values for each data point (rows) and each absolute
            feature (columns)

        :param sample: pd.DataFrame
            A pandas Dataframe that contains the input data, columns are the absolute features and each row
            contain the corresponding feature value

        :param pos: int
            The order of the feature in the feature importance of the model
            
        :param interaction: bool
            It decides where the shap dependency plot should include an interaction between features or not
            if True, interaction_index is set to 'auto', if false it set to None
            
        :param cmap: str
            A string used to define the color map used to draw the shap dependency plot with interactions

        :param interaction: bool
            It decides where the shap dependency plot should include an interaction between features or not
            if True, interaction_index is set to 'auto', if false it set to None

        :param cmap: str
            A string used to define the color map used to draw the shap dependency plot with interactions

        :param opacity: float
            The degree of the opacity of the scatter plot, if large data points use small opacity
            
        :param mean_na: bool
            Whether to include infromation about the mean shaply value for the feature when the value is
            not available (NA)

        :param mean_na: bool
            Whether to include infromation about the mean shaply value for the feature when the value is
            not available (NA)

        :param percentage: float
            The percentage of the given data that needed to be covered by the dependency plot

        :param y_label: str
            It defines what to be written on the Y-axis of the plot

        :param x_label: str
            It defines what to be written on the X-axis, either German, English or Raw feature name or
            any other given string

        :param label_len: int
            The maximum number of characters allowed per line for the x_label

        :param fdir: str
            The path to the directory where the plot is stored

        :param title_map: str
            It defines which mapping is used to name the title of the plot, default uses the raw feature name,
            you can either choose 'DE' for German mapping or 'EN' for  English mapping.

        :param title_len: int
            The maximum number of characters allowed per line for the title of the plot

        :param plot_format: str
            This defines the format used to save the plot '.png', '.jpg', '.pdf'

        :return None, It saves the requested plot on disk
        """

        _std, _mean, imp = self.num_feats_stats.loc[feat]
        shap_df_sel = shap_df[sample[feat] != imp]
        sample_sel = sample[sample[feat] != imp]

        fig, ax = plt.subplots(figsize=(5, 5))

        if len(shap_df_sel) > 0:
            _values = sample_sel[feat]
            _min, _max = _values.min(), _values.max()
            for scope in np.linspace(0.1, 3.0, num=30):
                x_min = max(_mean - scope * (_std + 10), _min - 1)
                x_max = min(_mean + scope * (_std + 10), _max + 1)
                if ((_values >= x_min) & (_values <= x_max)).sum() > len(_values) * percentage:
                    break

            if interaction:
                shap.dependence_plot(feat, shap_df_sel.values, sample_sel,
                                     cmap=matplotlib.cm.get_cmap(cmap),
                                     interaction_index='auto', show=False, ax=ax, alpha=opacity,
                                     xmin=x_min, xmax=x_max)

            else:
                shap.dependence_plot(feat, shap_df_sel.values, sample_sel,
                                     color=Config.colors['TRQ'],
                                     interaction_index=None, show=False, ax=ax, alpha=opacity,
                                     xmin=x_min, xmax=x_max)

        if (len(shap_df_sel) < len(shap_df)) and mean_na:
            mean_na_shap = shap_df[sample[feat] == imp][feat].mean()
            ax.text(1.2, -0.08, 'Mean \n Shap NA: \n %0.3f' % mean_na_shap, transform=ax.transAxes,
                    ha='center', va='center', fontsize=8, color='black')

        if (title_map == 'EN') and (feat in self.var_def_en):
            title = Viz.split_string_lines(self.var_def_en[feat], title_len)
            title = "Shap: %s \n" % title
        elif (title_map == 'DE') and (feat in self.var_def_de):
            title = Viz.split_string_lines(self.var_def_de[feat], title_len)
            title = "Shap: %s \n" % title
        else:
            title = "Shap: %s \n" % feat

        ax.set_title(title, fontsize=Config.TIT_FS, fontweight='bold')
        ax.set_ylabel(y_label, fontsize=Config.AXS_FS)

        if x_label in ['DE', 'EN', 'RF']:
            if (x_label == 'EN') and (feat in self.var_def_en):
                ax.set_xlabel(Viz.split_string_lines(self.var_def_en[feat], label_len),
                              fontsize=Config.AXS_FS)
            elif (x_label == 'DE') and (feat in self.var_def_de):
                ax.set_xlabel(Viz.split_string_lines(self.var_def_de[feat], label_len),
                              fontsize=Config.AXS_FS)
            else:
                ax.set_xlabel(feat, fontsize=Config.AXS_FS)
        else:
            ax.set_xlabel(x_label, fontsize=Config.AXS_FS)

        fig.savefig(os.path.join(fdir, '%03d_' % pos + feat + plot_format),
                    bbox_inches='tight')
        plt.close()

    def plot_shapely_values(self, ratio=0.05, cut_off=0.01, mean_na=True, opacity_num=0.5,
                            label_len=30, y_label_num='Effect on General_Churn', x_label_num='DE',
                            x_label_cat='Effect on General_Churn', y_label_cat='DE',
                            percent_num=0.99, cat_line_len=30, plot_type_cat='box',
                            dir_name='Shaply_Values', title_map='RAW', title_len=100,
                            plot_format='.pdf', interaction=True, cmap='autumn'):

        """This method is simply a rapper for the  for the two shapely values plotting functions used for
        categorical and numerical features. First the shap tree explainer is created then only a sample of
        the data given is drawn. The shap values are then created using this sample and finally, we go
        through the list of features in order of their importance and plot the shaply values

        :param ratio: float
            The percentage of the data for which the shaply values are calculated

        :param opacity_num: float
            The degree of the opacity of the scatter plot of the numerical shap dependency plot

        :param cut_off: float
            It defines the minimum mean shaply values for a category to be selected for the shaply
            values of categorical variable

        :param mean_na: bool
            Whether to show the mean shaply value of NA in numerical plots

        :param interaction: bool
            It decides where the shap dependency plot should include an interaction between features or not
            if True, interaction_index is set to 'auto', if false it set to None

        :param cmap: str
            A string used to define the color map used to draw the shap dependency plot with interactions

        :param y_label_num: str
            It defines what to be written on the Y-axis of the numerical plots

        :param x_label_num: str
            It defines what to be written on the X-axis of the numerical plots, it can be DE, EN, RF to
            choose between German, or English or raw feature names, if none of those were selected
            the given string will be used as the label on the X-axis

        :param x_label_cat: str
            It defines what to be written on the X-axis of the categorical plots

        :param y_label_cat: str
            It defines what to be written on the Y-axis of the categorical plots, it can be DE, EN, RF to
            choose between German, or English or raw feature names, if none of those were selected
            the given string will be used as the label on the X-axis

        :param label_len: int
            The maximum number of characters allowed per line for the x_label of the numerical plots
            and the y_label of the categorical plots

        :param percent_num: float
            The percentage of the given data that needed to be covered by the dependency plot of the
            numerical values

        :param cat_line_len: int
            The maximum number of characters allowed per line for the name of the categories appear on the y-axis
            of the categorical plots

        :param plot_type_cat: str
            It defines the type of the plot used for shaply values of categorical features 'box' or 'bar',
            default is 'bar'.

        :param dir_name: str
            The sub_directory under the viz_dir where all the shaply plots are stored

        :param title_map: str
            It defines which mapping is used to name the title of the plot, default uses the raw feature name,
            you can either choose 'DE' for German mapping or 'EN' for  English mapping.

        :param title_len: int
            The maximum number of characters allowed per line for the title of the plot

        :param plot_format: str
            This defines the format used to save the plot '.png', '.jpg', '.pdf'

        :return None, It saves the shaply values plots for the whole model on disk
        """

        if not os.path.exists(os.path.join(self.viz_dir, dir_name)):
            os.mkdir(os.path.join(self.viz_dir, dir_name))

        if not os.path.exists(os.path.join(self.viz_dir, dir_name, 'Numerical')):
            os.mkdir(os.path.join(self.viz_dir, dir_name, 'Numerical'))

        if not os.path.exists(os.path.join(self.viz_dir, dir_name, 'Categorical')):
            os.mkdir(os.path.join(self.viz_dir, dir_name, 'Categorical'))

        explainer = shap.TreeExplainer(self.model)
        sample = self.data_enc[np.random.choice(self.data_enc.shape[0],
                                                int(self.data_enc.shape[0] * ratio),
                                                replace=False), :]

        if isinstance(sample, scipy.sparse.csr.csr_matrix):
            sample = pd.DataFrame(sample.todense(), columns=self.abs_feats)
        elif isinstance(sample, np.ndarray):
            sample = pd.DataFrame(sample, columns=self.abs_feats)
        shap_df = pd.DataFrame(explainer.shap_values(sample), columns=self.abs_feats)

        fig = plt.figure()
        fig.suptitle('Shaply Values For Numerical Features', fontsize=20)
        shap.summary_plot(shap_df[self.raw_num_feats].values, sample[self.raw_num_feats],
                          show=False)

        fig.savefig(os.path.join(self.viz_dir, dir_name, 'Numerical' + plot_format),
                    bbox_inches='tight')

        plt.close()

        imp_df = self.get_feat_imp_df()
        imp_df.sort_values('VAR_IMP', ascending=False, inplace=True)
        imp_feats = imp_df['RAW_VAR'].values
        for i, feat in enumerate(imp_feats):

            if feat in self.raw_num_feats:
                self.plot_shap_numerical(feat, shap_df, sample, i, interaction, opacity_num, mean_na,
                                         percent_num, y_label_num, x_label_num, label_len,
                                         os.path.join(self.viz_dir, dir_name, 'Numerical'),
                                         title_map, cmap, title_len, plot_format)

            if feat in self.raw_cat_feats:

                self.plot_shap_categorical(feat, shap_df, i, cut_off, cat_line_len, plot_type_cat,
                                           y_label_cat, x_label_cat, label_len,
                                           os.path.join(self.viz_dir, dir_name, 'Categorical'),
                                           title_map, title_len, plot_format)

    def bin_score_deciles(self, fname='Deciles', bar=True, line=True, ylabel_bar='Probability',
                          xlabel_bar='Score Deciles', ylabel_line='True', xlabel_line='Predicted',
                          opacity=0.8, title='Deciles', plot_format='.pdf'):

        """This method is used to plot the performance of the predicted scores of the model verses the true label
        based on the deciles of the predictions. First the prediction scores are divided into 10 ranges (deciles),
        the mean of the true label and the prediction scores are calculated for each range and plotted

        :param fname: str
            The name of the file under which the plot is stored

        :param bar: bool
            Whether a bar plot is requested for the deciles or not

        :param line: bool
            Whether a line plot is requested for the deciles or not

        :param ylabel_bar: str
            It defines what to be written on the Y-axis of the bar plot

        :param xlabel_bar: str
            It defines what to be written on the X-axis of the bar plot

        :param ylabel_line: str
            It defines what to be written on the Y-axis of the line plot

        :param xlabel_line: str
            It defines what to be written on the X-axis of the line plot

        :param opacity: float
            The degree of the opacity of the bar plot

        :param title: str
            The title of the plot

        :param plot_format: str
            This defines the format used to save the plot '.png', '.jpg', '.pdf'

        :return None, It saves the requested plot on disk
        """

        df = pd.DataFrame({'CHURN_SCORE': self.pred_score, 'TRUE_SCORE': self.true_label})
        deciles = pd.qcut(df['CHURN_SCORE'], 10, duplicates='drop')
        df['SCORE_GROUP'] = deciles.values.codes
        df_graph = df.groupby(['SCORE_GROUP'])['CHURN_SCORE', 'TRUE_SCORE'].mean().reset_index()

        if bar:
            fig, ax = plt.subplots(figsize=(5, 5))
            ax = df_graph.plot(x='SCORE_GROUP', y='CHURN_SCORE', kind='bar', ax=ax, legend=False,
                               color=Config.colors['RED'], label='Predicted', alpha=opacity)
            ax = df_graph.plot(x='SCORE_GROUP', y='TRUE_SCORE', kind='bar', ax=ax, legend=False,
                               color=Config.colors['YEL'], label='True', alpha=opacity)
            ax.set_ylabel(ylabel_bar)
            ax.set_xlabel(xlabel_bar)
            ax.set_title(title + ' (bar)', fontsize=Config.TIT_FS, fontweight='bold')
            ax.legend(loc="best")
            fig.savefig(os.path.join(self.viz_dir, fname + '_bar' + plot_format),
                        bbox_inches='tight')
            plt.close()

        if line:
            fig, ax = plt.subplots(figsize=(5, 5))
            ax = df_graph.plot(x='CHURN_SCORE', y='TRUE_SCORE', ax=ax, legend=False,
                               color=Config.colors['RED'])
            ax.set_xlabel(xlabel_line)
            ax.set_ylabel(ylabel_line)
            ax.set_title(title + ' (line)', fontsize=Config.TIT_FS, fontweight='bold')
            fig.savefig(os.path.join(self.viz_dir, fname + '_line' + plot_format),
                        bbox_inches='tight')
            plt.close()

    def plot_pred_distribution(self, fname='Prediction_distribution', classes=("STAY", "CHURN"),
                               bins=100, opacity=0.8, denisty=True, ylabel='Frequency',
                               xlabel='Probability', title='Conditional Distribution Predictions',
                               plot_format='.pdf'):

        """This method plot the probability distribution of the predicted scores conditioned on the true
        labels for the given data.

        :param fname: str
            The name of the file under which the plot is stored

        :param classes: tuple
            It is a tuple of strings that contains the name of the binary classes

        :param bins: int
            number of bins used to plot the histogram of the probability distributions

        :param opacity: float
            The degree of the opacity of the plots (distributions overlap)

        :param denisty: bool
            Whether the histogram is count or denisty

        :param ylabel: str
            It defines what to be written on the Y-axis of the plot

        :param xlabel: str
            It defines what to be written on the X-axis of the plot

        :param title: str
            The title of the plot

        :param plot_format: str
            This defines the format used to save the plot '.png', '.jpg', '.pdf'

        :return None, It saves the requested plot on disk
        """

        df = pd.DataFrame({'CHURN_SCORE': self.pred_score, 'TRUE_SCORE': self.true_label})
        stay = df[df['TRUE_SCORE'] == 0]['CHURN_SCORE']
        churn = df[df['TRUE_SCORE'] == 1]['CHURN_SCORE']

        fig, ax = plt.subplots(figsize=(5, 5))
        ax = stay.plot.hist(bins=bins, color=Config.colors['RED'], ax=ax, density=denisty,
                            alpha=opacity)
        ax = churn.plot.hist(bins=bins, color=Config.colors['YEL'], ax=ax, density=denisty,
                             alpha=opacity)

        ax.set_title(title, fontsize=Config.TIT_FS, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=Config.AXS_FS)
        ax.set_xlabel(xlabel, fontsize=Config.AXS_FS)
        ax.tick_params(axis='y', which='major', labelsize=Config.TIC_FS)
        ax.legend(classes, loc='best')
        fig.savefig(os.path.join(self.viz_dir, fname + plot_format), bbox_inches='tight')
        plt.close()
