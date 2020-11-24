Config\_example module
============================
In the same folder for all projects the following files are requested:


abt_vars.txt
------------------------
file containing all features that are going to be used in the model


population_scoring.sql
------------------------
Sql query to obtain the scoring dataset from snowflake

population_train.sql
------------------------
Sql query to obtain the train dataset from snowflake

config_file.yaml
------------------------

The config file is divided in 4 parts:

1. Common part for all steps:

.. code-block:: bash

    user: ''         #
    sql_extract: False     #extract data from snowflake based on the following key (train,
    scoring)
    train: True            # apply the train
    scoring: False         # apply scoring if the model have been already trained
    visual: True           # apply visualization if the model have been trained
    score_name: 'General_Churn'   # string with model (ex. General_Churn, Price_increase)
    main_dir: '../..'             # path to main dir to store input and output data
    abt_vars: 'abt_vars.txt'      # name of txt file containing the feature list
    target: 'con_churn_365_flg'   # target column


2. Data extraction specific

.. code-block:: bash

    extraction:        # snowflake extraction
      sample: 300000
      analytics_base_table: 'AE_ANALYTICS_DM.ANALYTICS_DM_T_BASE'
      population_train:
        name: "pop_training"
        file: "population_train.sql"
        join_type: 'left'
        join_pop:
            - - "CONTRACT_ID"
              - "="
              - "CONTRACT_ID"
            - - "FULL_DT"
              - "="
              - 'FULL_DT'
      population_score:
        name: "pop_score"
        file: "population_scoring.sql"
        join_type: 'left'
        join_pop:
            - - "CONTRACT_ID"
              - "="
              - "CONTRACT_ID"
            - - "FULL_DT"
              - "="
              - 'FULL_DT'

3. Model Specific:

.. code-block:: bash

    prediction:
      random_split: False                     # if True getting Random split of train 0.6 test 0.2 eval 0.2
      last_date_test: '2018-01-31'            # smaller 'full_dt' considered on test: test=( df['full_df']>=last_date_test )
      ratio_weight: 0.04864                   # weight to unbalance the train set
      param:                                  # model parameters for cross validation
            'objective': 'binary:logistic'
            'verbosity': 1
            'booster': 'gbtree'
            'n_jobs': 2
            'max_depth': 2
            'n_estimators': [100, 200, 300]
            'learning_rate': 0.2
            'reg_lambda': 3
            'subsample': 0.75
            'random_state': 1337
      num_imputer: -99999                                    # Value for imputation of numerical variables
      features: []                                           # Possibility to list specific features during the training

4. Visualization Specific

.. code-block:: bash

    vis:
      var_def: 'visualization/VAR_DEF.csv'       # path to variable definition
      auc:                                                              # Area Under the Curve
        flg: True
        param:
          plot_size: 5
          fname: 'AUC_ROC'
          table: True
          save_format: 'csv'
          plot_format: '.png'

      imp:                                                             # feature importance
        t_flg: True
        t_param:
          aggregated: True
          save_format: 'csv'
          fname : 'Feature_Importance'

        p_flg: True
        p_param:
          aggregated: True
          x: ['RAW_VAR', 'VAR_DEF_EN', 'VAR_DEF_DE']
          y: 'VAR_IMP_REL'
          x_label: 'Model Features'
          y_label: 'Relative Importance'
          fname: 'Feature_Importance_'
          str_len: [25, 25, 35]
          plot_format: '.png'

      em:                                                              # evaluation metric
        flg: True
        param:
          plot_size: 8
          fname: 'ACC_PRE_REC_F1'
          table: True
          save_format_table: 'csv'
          metrics: ['ACC', 'PRE', 'REC', 'F1']
          plot_format: '.png'

      cf_mat:                                                         # confusion metric
        flg: True
        param:
          threshold: ['None', 'None', 'None', 'None']
          plot_size: 10
          classes: ['Stay', 'General_Churn']
          normalize_axis: ['None', 'pred', 'all', 'true']
          cmap: Reds
          fname: 'Confusion_Matrix_'
          plot_format: '.png'

      shap:                                                           # Shaply values
        flg: True
        param:
          ratio: 0.05
          cut_off: 0.01
          opacity_num: 0.5
          percent_num: 0.95
          cat_line_len: 30
          plot_type_cat: 'bar'
          label: 'Effect on General Churn'
          dir_name: 'Shaply_Values'
          title_map: 'DE'
          title_len: 35
          plot_format: '.png'

      decile:
        flg: True
        param:
          bar: True
          line: True
          ylabel_bar: 'Probability'
          xlabel_bar: 'Score Deciles'
          xlabel_line: 'Predicted Probability'
          ylabel_line: 'True Probability'
          opacity: 0.8
          title: 'Deciles'
          plot_format: '.png'

      dist:
        flg: True
        param:
          fname: 'Prediction_distribution'
          classes: ["STAY", "CHURN"]
          bins: 100
          opacity: 0.8
          denisty: True
          title: 'Conditional Distribution Predictions'
          ylabel: 'Frequency'
          xlabel: 'Probability'
          plot_format: '.png'