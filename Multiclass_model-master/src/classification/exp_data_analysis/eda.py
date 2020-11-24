import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class Config:
    """ A Configuration class that contains some constants used during visualization"""
    colors = {'TRQ': (0.36, 0.76, 0.8), 'RED': (0.92, 0.11, 0.04),
              'YEL': (0.89, 0.88, 0), 'BOR': (0.69, 0.02, 0.05)}
    MAX_STR_LEN = 35
    TIT_FS = 15
    AXS_FS = 12
    TIC_FS = 8
    FONT = {'family': 'Arial', 'weight': 'normal', 'size': 8}


matplotlib.rc('font', **Config.FONT)


class Eda:
    """ A Result visualization class built for tree based machine learning models.
       the visualizations are based on SHAP along with some basic visualization plots.


    :param     data: pd.DataFrame
            A pandas DataFrame for the data obtained from the train sql. The DataFrame contains
            all the columns downloaded.

    :param    features: list
            It contains a list of the names of the features used by the model.

    :param     target: str
            It contains the names of the target variable

    :param     eda_dir: str
            A string for the directory that will be used to store the plots

    :param     var_def_file: str
            A path to a csv file that contain the English and German explanations of each
            of the features used by the model

        Attributes
        ----------
        num_feats: list
            A list of the names of the numerical features

        cat_feats: list
            A list of the names of the categorical features

        var_def_en: dict
            A dictionary the store the English explanation for each raw feature

        var_def_de: dict
            A dictionary the store the German explanation for each raw feature
        """

    def __init__(self, data, features, target, classes, eda_dir, var_def_file):
        self.data = data[features + [target]].copy()
        self.features = features
        self.target = target
        self.classes = classes

        self.num_feats = data[features].select_dtypes(include=np.number).columns
        self.cat_feats = data[features].select_dtypes(include=object).columns

        self.var_def = pd.read_csv(var_def_file)
        #self.var_def['RAW_VAR'] = self.var_def['RAW_VAR'].str.lower()
        self.var_def_en = dict(zip(self.var_def['RAW_VAR'], self.var_def['VAR_DEF_EN']))
        self.var_def_de = dict(zip(self.var_def['RAW_VAR'], self.var_def['VAR_DEF_DE']))

        self.eda_dir = eda_dir

    @staticmethod
    def split_string_lines(input_str, max_len=Config.MAX_STR_LEN):

        """This method splits the input string into lines such that the max number of characters
        in each line is determined by max_len. It is important to keep in mind that the split of
        the input string is done based on spaces.

        :param input_str: str
            input string to be split Ex:"A Group of Data for each line"

        :param max_len: int
            the max length allowed in each line, Ex: 10

        :return It return a new string split into lines: "A Group of \n Data for \n each line" .
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

    def plot_numerical(self, percentage=0.9, opacity=0.5, colors=['RED', 'YEL'], denisty=False,
                       title='DE', title_len=60, x_label='DE', y_label='Count', label_len=35,
                       add_stats=True, plot_format='.pdf'):

        """This method is used to plot all numerical features against the target variable. The plot is a
        histogram, in which the data given is split into the two classes of the target variable. For each
        class a histogram of the data points is plotted. This plot help to understand how a given feature
        affect the class of the target variable.

        :param percentage: float
            The percentage of the given data that needed to be covered by the plot, in general we select
            the data points around the mean until we have the required percentage

        :param opacity: float
            The degree of the opacity of the histograms because they usually overlaps

        :param colors: list
            This selects the colors to be used for each class, it is a list of strings where we
            only have 4 possible colors: ['TRQ', 'RED', 'YEL', 'BOR']

        :param denisty: bool
            It is a boolean flag to determine whether the histograms bins are plotted to
            represent denisty or actual data points count

        :param title: str
            It defines which mapping is used to name the title of the plot, you can either choose 'DE'
            for German mapping or 'EN' for English mapping, or 'RF' to use the raw feature name. If
            none of the previous three options were selected, the title is set to the given string.

        :param title_len: int
            The maximum number of characters allowed per line for the title of the plot only when
            'DE' or 'EN' are given as title

        :param x_label: str
            It defines what to be written on the X-axis, either German, English or Raw feature name
            when the input is 'DE', 'EN', 'RF' respectively or the given string otherwise

        :param y_label: str
            It defines what to be written on the Y-axis of the plot

        :param label_len: int
            The maximum number of characters allowed per line for the x_label, again this is only
            effective if x_label is 'DE' or 'EN'

        :param add_stats: bool
            This is a boolean flag the determines whether some statistics like the mean of each class
            of the target variable as well as the NA count to be written on the plot

        :param plot_format: str
            This defines the format used to save the plot '.png', '.jpg', '.pdf'

        :return None, It saves the requested plots on disk
        """

        p_dir = os.path.join(self.eda_dir, 'Numerical')
        if not os.path.exists(p_dir):
            os.mkdir(p_dir)

        for feat in self.num_feats:
            _df = self.data[[feat, self.target]].copy()
            na_per = _df[feat].isna().mean()
            _df = _df[~_df[feat].isna()].copy()

            _mean, _std = _df[feat].mean(), _df[feat].std()
            _min, _max = _df[feat].min(), _df[feat].max()

            for scope in np.linspace(0.01, 3.0, num=300):
                x_min = max(_mean - scope * (_std + 10), _min - 1)
                x_max = min(_mean + scope * (_std + 10), _max + 1)
                if ((_df[feat] >= x_min) & (_df[feat] <= x_max)).sum() > len(_df) * percentage:
                    break

            _df = _df[(_df[feat] >= x_min) & (_df[feat] <= x_max)].copy()
            tmp_df = pd.DataFrame()
            labels = np.sort(_df[self.target].unique())

            for label in labels:
                tmp_df[label] = _df[_df[self.target] == label][feat].reset_index(drop=True)

            fig, ax = plt.subplots(figsize=(5, 5))
            ax = tmp_df.plot.hist(bins=min(max(_df[feat].nunique()//100, 10), 50), density=denisty,
                                  ax=ax, alpha=opacity, legend=False,
                                  color=[Config.colors[c] for c in colors])

            if x_label in ['DE', 'EN', 'RF']:
                if (x_label == 'EN') and (feat in self.var_def_en):
                    ax.set_xlabel(Eda.split_string_lines(self.var_def_en[feat], label_len),
                                  fontsize=Config.AXS_FS)
                elif (x_label == 'DE') and (feat in self.var_def_de):
                    ax.set_xlabel(Eda.split_string_lines(self.var_def_de[feat], label_len),
                                  fontsize=Config.AXS_FS)
                else:
                    ax.set_xlabel(feat, fontsize=Config.AXS_FS)
            else:
                ax.set_xlabel(x_label, fontsize=Config.AXS_FS)

            ax.set_ylabel(y_label, fontsize=Config.AXS_FS)

            if add_stats:
                for l in labels:
                    i = np.where(np.array(self.classes) == l)[0][0]
                    ax.text(1.15, 0.08*(i+1), 'Mean for %s: \n %0.3f' % (l, tmp_df[l].mean()),
                            transform=ax.transAxes, ha='center', va='center', fontsize=8, color='black')

                ax.text(1.15, 0, 'NA Percentage: \n %0.3f' % na_per, transform=ax.transAxes,
                        ha='center', va='center', fontsize=8, color='black')

            ax.legend(loc='upper left', bbox_to_anchor=(1, 1), ncol=1)

            if title in ['DE', 'EN', 'RF']:
                if (title == 'EN') and (feat in self.var_def_en):
                    tit = 'Distribution of %s among the different classes' % self.var_def_en[feat]
                    ax.set_title(Eda.split_string_lines(tit, title_len), fontsize=Config.TIT_FS,
                                 fontweight='bold')
                elif (title == 'DE') and (feat in self.var_def_de):
                    tit = 'Verteilung von %s auf die verschiedenen Klassen' % self.var_def_de[feat]
                    ax.set_title(Eda.split_string_lines(tit, title_len), fontsize=Config.TIT_FS,
                                 fontweight='bold')
                else:
                    ax.set_title(feat, fontsize=Config.TIT_FS, fontweight='bold')
            else:
                ax.set_title(title, fontsize=Config.TIT_FS, fontweight='bold')

            fig.savefig(os.path.join(p_dir, feat + plot_format), bbox_inches='tight')
            plt.close()

    def plot_categorical(self, percentage=0.01, plot_type='barh', color='RED', column='Stay',
                         balance=0, rot=0, denisty=True, title='DE', title_len=60, x_label='DE',
                         y_label='Count', label_len=35, cat_len=30, fig_size=5, plot_format='.pdf'):

        """This method is used to plot all categorical features against the target variable. The plot is a
        bar plot where for each category of the categorical feature, the bar represents the number of data
        points that belongs to each class of the target variable. This plot help to understand how a
        given categorical feature affect the class of the target variable.

        :param percentage: float
            The minimum percentage of the given data that a category needs to contain in order to
            be considered in the plot.

        :param plot_type: str
            This defines the type of the plot, we only allow 'bar' and 'barh'.

        :param color: str
            This selects the color to be used for the bar, we only have 4 possible colors:
            ['TRQ', 'RED', 'YEL', 'BOR']

        :param column: str
            The name of the target class for which the plot is done, it can take values from the
            self.classes list

        :param balance: float
            The ratio needed to re balance the numbers of the contracts to adjust for the fact that
            the given data set is balanced, although the original is not. This ratio is multiplied by
            the values of the class given in column, so make sure you adjust it right.

        :param rot: int
            This is an integer to define the angle by which the text of the categories is rotated.
            we advice to use rot=0 for 'barh' and rot=90 for 'bar'

        :param denisty: bool
            It is a boolean flag to determine whether the bars for each category are plotted to
            represent the actual data points count or to be normalized for each category

        :param title: str
            It defines which mapping is used to name the title of the plot, you can either choose 'DE'
            for German mapping or 'EN' for English mapping, or 'RF' to use the raw feature name. If
            none of the previous three options were selected, the title is set to the given string.

        :param title_len: int
            The maximum number of characters allowed per line for the title of the plot only when
            'DE' or 'EN' are given as title

        :param x_label: str
            It defines what to be written on the X-axis, either German, English or Raw feature name
            when the input is 'DE', 'EN', 'RF' respectively or the given string otherwise

        :param y_label: str
            It defines what to be written on the Y-axis of the plot

        :param label_len: int
            The maximum number of characters allowed per line for the x_label, again this is only
            effective if x_label is 'DE' or 'EN'

        :param cat_len: int
            The maximum number of characters allowed per line for the each category

        :param fig_size: int
            It defines the size of each plot, it it is set to 0, the size will be calculated based on
            the number of the categories and the plot_type

        :param plot_format: str
            This defines the format used to save the plot '.png', '.jpg', '.pdf'

        :return None, It saves the requested plots on disk
        """

        p_dir = os.path.join(self.eda_dir, 'Categorical')
        if not os.path.exists(p_dir):
            os.mkdir(p_dir)

        for feat in self.cat_feats:
            _df = self.data[[feat, self.target]].copy()
            data_points = len(_df)
            _df[feat] = np.where(_df[feat].isna(), 'UNKNOWN', _df[feat])

            _df = _df.groupby([feat])[self.target].value_counts().unstack(level=[1])
            _df = _df.reset_index().fillna(0)
            #_df.columns = [feat] + [self.classes[int(i)] for i in _df.columns[1:]]
            _df.columns = [feat] + [i for i in _df.columns[1:]]

            _df['TOTAL'] = _df[_df.columns[1:]].sum(axis=1)
            _df = _df[_df['TOTAL'] >= percentage * data_points]

            if balance:
                _df[column] = _df[column] * balance
                _df['TOTAL'] = _df[_df.columns[1:]].sum(axis=1)

            if denisty:
                _df[_df.columns[1:]] = _df[_df.columns[1:]].div(_df['TOTAL'], axis=0)
                _df[_df.columns[1:]] = _df[_df.columns[1:]] * 100

            if len(_df) > 0:

                if fig_size:
                    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
                elif plot_type == 'barh':
                    fig, ax = plt.subplots(figsize=(5, 0.8 * len(_df)))
                else:
                    fig, ax = plt.subplots(figsize=(0.8 * len(_df), 5))

                _df.drop('TOTAL', axis=1, inplace=True)
                _df[feat] = np.vectorize(Eda.split_string_lines)(_df[feat], cat_len)

                _df.plot(x=feat, y=column, kind=plot_type, ax=ax, legend=False, rot=rot,
                         color=Config.colors[color])

                if plot_type == 'bar':
                    if x_label in ['DE', 'EN', 'RF']:
                        if (x_label == 'EN') and (feat in self.var_def_en):
                            ax.set_xlabel(Eda.split_string_lines(self.var_def_en[feat], label_len),
                                          fontsize=Config.AXS_FS)
                        elif (x_label == 'DE') and (feat in self.var_def_de):
                            ax.set_xlabel(Eda.split_string_lines(self.var_def_de[feat], label_len),
                                          fontsize=Config.AXS_FS)
                        else:
                            ax.set_xlabel(feat, fontsize=Config.AXS_FS)
                    else:
                        ax.set_xlabel(x_label, fontsize=Config.AXS_FS)

                    ax.set_ylabel(y_label, fontsize=Config.AXS_FS)

                elif plot_type == 'barh':
                    if x_label in ['DE', 'EN', 'RF']:
                        if (x_label == 'EN') and (feat in self.var_def_en):
                            ax.set_ylabel(Eda.split_string_lines(self.var_def_en[feat], label_len),
                                          fontsize=Config.AXS_FS)
                        elif (x_label == 'DE') and (feat in self.var_def_de):
                            ax.set_ylabel(Eda.split_string_lines(self.var_def_de[feat], label_len),
                                          fontsize=Config.AXS_FS)
                        else:
                            ax.set_ylabel(feat, fontsize=Config.AXS_FS)
                    else:
                        ax.set_ylabel(x_label, fontsize=Config.AXS_FS)

                    ax.set_xlabel(y_label, fontsize=Config.AXS_FS)

                else:
                    break

                if title in ['DE', 'EN', 'RF']:
                    if (title == 'EN') and (feat in self.var_def_en):
                        tit = 'Distribution of the classes among the different categories of %s' \
                              % self.var_def_en[feat]
                        ax.set_title(Eda.split_string_lines(tit, title_len), fontsize=Config.TIT_FS,
                                     fontweight='bold')
                    elif (title == 'DE') and (feat in self.var_def_de):
                        tit = 'Verteilung von Klassen auf die verschiedenen Kategorien von %s' \
                              % self.var_def_de[feat]
                        ax.set_title(Eda.split_string_lines(tit, title_len), fontsize=Config.TIT_FS,
                                     fontweight='bold')
                    else:
                        ax.set_title(feat, fontsize=Config.TIT_FS, fontweight='bold')
                else:
                    ax.set_title(title, fontsize=Config.TIT_FS, fontweight='bold')

                plt.xticks(fontsize=Config.TIC_FS)
                plt.yticks(fontsize=Config.TIC_FS)

                fig.savefig(os.path.join(p_dir, feat + plot_format), bbox_inches='tight')
                plt.close()
