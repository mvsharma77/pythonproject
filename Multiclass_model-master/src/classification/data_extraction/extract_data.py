"""
Functions to extract data sets form snowflake.
"""

import os
import logging
import pandas as pd


def create_table(sql_file, table_name, engine):
    """create table in snowflake or re-populate the table if already exists.

    :param sql_file: str
        path to the sql file t create the table
    :param table_name: str
        table name for which we want to create the table
    :param engine: object
        snowflake connector
    """

    with open(sql_file, 'r') as my_file:
        sql_str = my_file.read()

    # create table if not exists
    sql_str_create = 'create table if not exists ' + table_name + ' as ' + sql_str
    sql_text = sql_common.convert_str_2_text(sql_str_create)
    logging.info('creating table %s', table_name)
    result = engine.execute(sql_text)
    logging.info(result.fetchall())
    logging.info('completed table %s', table_name)

    # clean the table
    sql_str_truncate = 'truncate table ' + table_name
    sql_text = sql_common.convert_str_2_text(sql_str_truncate)
    logging.info('Truncate table %s', table_name)
    result = engine.execute(sql_text)
    logging.info(result.fetchall())

    sql_str_insert = 'insert into ' + table_name + sql_str
    sql_text = sql_common.convert_str_2_text(sql_str_insert)
    logging.info('Truncate table %s', table_name)
    result = engine.execute(sql_text)
    logging.info(result.fetchall())


def create_view(population_table, population_config, features_file, config, view_name, engine,
                train=True):
    """
    create view in snowflake based on the following parameter

    :param population_table: str table's name
    :param population_config: dict with configuration paths
    :param features_file: str  path of the feature file
    :param config: class     configuration class
    :param view_name: str     name of the view
    :param engine: object    snowflake connector
    :param train: bool
    """

    pop_alias = 'pop'
    abt_alias = 'abt'

    abt_vars = get_abt_vars(features_file)
    select_pop = create_select(pop_alias, ['*'])
    select_abt = create_select(abt_alias, abt_vars)
    if train:
        random_sort_str = ''' row_number() over(partition by {} order by random()) as 
                          random_sort'''.format(config.target)
        select_sql = 'select ' + ', '.join([select_pop, select_abt, random_sort_str])
    else:
        select_sql = 'select ' + ', '.join([select_pop, select_abt])

    abt_table = config.extraction['analytics_base_table']

    population_joins = population_config['join_pop']
    join_type = population_config['join_type']

    from_join = create_from_join(pop_alias, population_table, abt_alias, abt_table,
                                 population_joins, join_type)

    drop_str = 'drop view if exists {} '.format(view_name)
    sql_text = sql_common.convert_str_2_text(drop_str)
    engine.execute(sql_text)
    create_str = 'create view {} as '.format(view_name)
    create_view_str = '\n'.join([create_str, select_sql, from_join])
    sql_text = sql_common.convert_str_2_text(create_view_str)
    logging.info('trying to create view %s', sql_text)

    result = engine.execute(sql_text)
    logging.info('View created %s', result.fetchall())


def create_from_join(pop_alias, pop_table, abt_alias, abt_table, pop_joins, join_type):
    """
    create select statement to join the population's tables

    :param pop_alias: str                   table name

    :param pop_table: str
        population table's name

    :param abt_alias:  str
            list of features to be filtered out

    :param abt_table: str
            list of features to be filtered out

    :param pop_joins:  str
                abt population's name

    :param join_type: str
               type of join

    :return: str
    """

    from_str = ' '.join(['from', pop_table, pop_alias])
    join_start = ' '.join([join_type, 'join', abt_table, abt_alias, 'on'])
    join_filters = ' and '.join([format_join(pop_alias, join[0], abt_alias, join[2], join[1])
                                 for join in pop_joins])

    return ' \n'.join([from_str, join_start, join_filters])


def create_select(table_alias, variable_list):
    """
    create select statement t filter a list of columns

    :param table_alias: str                   table name
    :param variable_list: str                    list of features to be filtered out
    :return: str
    """

    select_list = [table_alias+'.'+var for var in variable_list]
    select_str = ', '.join(select_list)
    return select_str


def get_abt_vars(variables_file):
    """
    Extract list of variable from the a text file given as input

    :param variables_file: str       path to the file abt_vars.txt with list of features

    :return : list of string with variables names
    """

    with open(variables_file, 'r') as my_file:
        vars_selected = my_file.read().replace('\n', '').split(',')

    return vars_selected


def format_join(left_table_alias, left_join_column, right_table_alias, right_join_column,
                eval_function):

    """
    create sql statement to join two ables in snowflake

    :param left_table_alias: str    name of left table to be joined
    :param left_join_column: str    name of joining key column names for left join table
    :param right_table_alias: str   name of right table to be joined
    :param right_join_column: str   name of joining key column names for right join table
    :param eval_function: sql function
    :return: sql statement
    """

    left_column = '.'.join([left_table_alias, left_join_column])
    if isinstance(right_join_column, str):
        right_column = '.'.join([right_table_alias, right_join_column])
    else:
        right_column = ' and '.join([right_table_alias+'.' + col for col in right_join_column])

    joined = ' '.join([left_column, eval_function, right_column])
    return '('+joined+')'


def main(config, train=True):
    """
    data extraction from snowflake based on the sql

    :param config: class            with configuration paths
    :param train: bool if True wil extract the dataset for the training
           if False wil extract the dataset for the Scoring
    :return: store file in set folder
    """

    if train:
        logging.info('running the training')
        pop_config = config.extraction['population_train']
        population_file = os.path.join(config.data_input, pop_config['file'])
        data_set_path = os.path.join(config.data_dir, config.score_name + '_train_extract.pkl')


    else:
        logging.info('running the scoring')
        pop_config = config.extraction['population_score']
        population_file = pop_config['file']

    final_df = pd.read_csv(population_file)
    final_df['TotalCharges'] = pd.to_numeric(final_df['TotalCharges'], errors='coerce')
    if not os.path.exists(config.data_dir):
        os.makedirs(config.data_dir)

    data_set_path = os.path.join(config.data_dir, config.score_name + '_train_extract.pkl')


    final_df.to_pickle(data_set_path)

    logging.info('data_correctly extracted at path %s', data_set_path)
