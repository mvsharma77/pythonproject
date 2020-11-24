.. General Model documentation master file, created by
   sphinx-quickstart on Wed Oct 23 11:47:46 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

===========================================================
Welcome to General Classification Model's documentation!
===========================================================
   This project aims to create a classification pipeline based on a a data set extracted from
   Snowflake.

   The pipeline is divided in three steps:
    1. Data set extraction from SnowFlake
    2. Machine learning Model train/ scoring (features encoding, imputation, Xgboos)
    3. Visualization of the trained model based on the test dataset divided in a temporal or random split from the training data set.

   During the training the model applies a fix split between train test and eval applying
   cross-validation based on the parameters given in the configuration file.
   In addition it is possible to ge the prediction on the pre-trained Xgboost model.



   The code of the project is on Github:
   `Classification-Model `_

Features of General Classification Model
===========================================

* `Configuration Example`_, default configurations needed to run the classification package
* `Data Extraction`_, there is a module dedicated to data extraction from Snowflake. From the
  configuration file it is possible to extract train or scoring data based on the sql file given
  as input.
* `Train`_, The model will be trained with the specific train data
  defined in the config file.  Train data will be divided in train, validation, test set based
  on  random or temporal split. A cross_validation step will be applied during the training and
  if a range of parameters are given will provide the best model based on the validation set
  performance.
* `Scoring`_, in the model a module will be dedicated to score the dataset.
* `Visualization`_, in the library an entire module is dedicated on the creation of
  plots in order to interprete the new model performances.

.. _install_sg:

Installation
============

Installation via pip
------------------------
In order to install the package on your machine:

.. code-block:: bash

    $ git clone
    $ cd ddd-ml-pipeline
    $ pip install .

Install as a developer
------------------------
To install all package follow the steps below:

.. code-block:: bash

    $ git clone
    $ cd ddd-ml-pipeline
    $ pip install -r requirements.txt
    $ pip install -e .


Before running check the file ***config.yaml***
and be sure that the following are set to true since you will not have any data yet.

.. code-block:: bash

   train: True
   sql_extract: True


Move to the main repo and execute:

.. code-block:: bash

   > cd run
   > python main.py -user KID

If you are not running from the folder main you need to specify the path to the
config.yaml as an argument:

.. code-block:: bash

   > python main.py -cf (path to config.yam) -user KID


You will also need to install the dependencies listed above and `pytest`

How To run it:
=================

   In order to run the pipeline the configurations files need to be all in the same folder:

    - abt_vars.txt
    - population_scoring.sql
    - config.yml
    - population_train.sql

   After doing it and adjusting the config.yml with your requirements and move into Python:

.. code-block:: bash

    >> from classification import run_pipeline as run
    >> run.run_pipeline('path_to_folder_containing_yml', 'your KID')

If you want to run it from terminal:

.. code-block:: bash

    $ cd ddd-ml-pipeline\src\classification
    $ python run_pipeline.py -cf 'path_to_folder_containing_yml' -user 'your KID'

Configuration Example
======================

.. toctree::
   :maxdepth: 3
   :caption: Configuration files:

   configs


Data Extraction
==================
.. toctree::
   :maxdepth: 3
   :caption: Extraction step:

   data_extraction

Train
===========

.. toctree::
   :maxdepth: 3
   :caption: Train

   model_train

Scoring
====================================

.. toctree::
   :maxdepth: 3
   :caption: Scoring module

   model_score

Visualization
================

.. toctree::
   :maxdepth: 3
   :caption: Creation of plots

   visualization



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`





















