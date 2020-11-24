# Generic Model

This project aims to create a classification pipeline based on a a data set extracted from the file stored locally.
The pipeline is divided in four steps:
 1. Data set extraction from local repository
 2. Exploration Data Analysis
 3. Machine learning Model train/ scoring (features encoding, imputation, Xgboost)
 4. Visualization of the trained model based on the test dataset divided in a temporal or random 
 split from the training data set.
 
During the training the model applies a fix split between train test and eval applying 
cross-validation based on the parameters given in the configuration file.
In addition it is possible to ge the prediction on the pre-trained Xgboost model.

## Getting Started

These instructions will get you a copy of the project up and running 
on your local machine for development and testing purposes. 
See deployment for notes on how to deploy the project on a live system.

### Prerequisites

You need Python 3.7.3 or later to run this library.

In Ubuntu, Mint and Debian you can install Python 3 like this:
```
$ sudo apt-get install python3 python3-pip
```

For other Linux flavors, macOS and Windows, packages are available at

http://www.python.org/getit/

### Installing via Pip
In order to install the library:
```
> pip install
```
### Installing via git
For a development mode:
clone the repo [GitHub Page]() and
install the requirements.
After that run the setup.py to create the executable files.
```
> git clone (page)
> cd MULTICLASS_MODEL
> pip install -r requirements.txt
> python setup.py build
> python setup.py install

```

**Required structure folder**

In order to run the model you would need a folder containing all the configurations needed:
 -config_file.yaml  (configuration file)
 -abt_tars.txt (list of features)
 -population_scoring.sql (sql for scoring dataset)
 -population_training.sql (sql for training dataset)
 -var_def.csv (descriptions of the variables) 

**Set up config_file.yaml file**

Before running check the file ***config_file.yaml***
and be sure that the following are set to true since you will not have any data yet.
````
sql_extract_train: True
sql_extract_scoring: True
file: "/path-of-the-file-hosted-locally"
````

### run pipeline
````
Multiclass_model>> python -m classification.run_pipeline -dp .

````
### Run from Python console (optional)

After installing the library get into python
```
> python
>> from classification.run_pipeline import run_pipeline as rp
>> rp('yml path', "data_path")
```

### Runing from terminal (optional)

After installing the library it is possible to see the configurations and to run the pipeline autmatically.
To run the pipeline you can just run the following command
```
> ml_run_pipeline -cp -dp
```
with the options -cp it is possible to specify the path to the configuration yaml file
with -dp the path to the input data if not the default one.

To check how the configuration yaml file has to look like run: 
```
> ml_show_config
```


### Configuration.yaml

Before running check the file ***config.yaml***
and be sure that the following are set to true since you will not have any data yet.
```
sql_extract_train: True
sql_extract_scoring: True
```

## Running the tests

To run the test from the main folder:

````
 python -m unittest discover -s src
````

## Structure

The project follow the structure below:

```bash
├── src                     # folder containing library and configurations samples:
│   ├── classification          # contains the classification library  
│   │   ├── data_extraction          # contains the functions to query Snowflake and providing input data for the next steps 
│   │   │   └── extract_data.py            
│   │   │
│   │   ├── exp_data_analysis          # contains the functions to plot the analysis of input datasets
│   │   │   ├── eda.py 
│   │   │   └── main_eda.py 
│   │   │
│   │   ├── model          # model pipeline  
│   │   │   ├── create_pip.py        # create the scikit learn pipeline to encode the features and apply the model.
│   │   │   ├── fit_pipeline.py      # split train test eval and train the model
│   │   │   ├── grid_search.py       # apply grid search on parameters
│   │   │   ├── preprocessing.py     # fill null with nan and select features
│   │   │   ├── scoring_predict.py   # score with the trained model
│   │   │   ├── train.py             # main train script
│   │   │   ├── transformers.py      # custom transformers
│   │   │   └── utils.py  
│   │   │
│   │   ├── visualization          # functions to produce all the plots that are needed to view the trained model
│   │   │   ├── Viz.py 
│   │   │   └── main_viz.py 
│   │   │
│   │   ├── tests          
│   │   │   ├── test_model.py 
│   │   │   └── test_pipeline.py
│   │   │
│   │   ├── main_config.py              # main default config class used to force the paths and to get main variables to be used in the all project
│   │   └── run_pipeline.py             #  main script to run the full pipeline
│   │
│   └── configs                     # contains all the scripts needed fot the configurations
│       ├── abt_vars.txt            # list of features that will be use in the model
│       ├── population_scoring.sql  # sql to get the data set for scoring purposes
│       ├── population_train.sq     # sql to get the data set for training and splitting it in train, eval, test
│       ├── config_file.yml              # configuration file
│       └── var_def.csv             # description of features definition
│   
├── README.md
├── requirements.txt
├── setup.py
└── .gitignore
```

## Workflow and config.yml options

The flow is basically divided in five steps:

1. Data Extraction
2. Data Exploration
3. Modeling (Train)
4. Visualization
5. Scoring

The steps are sequential (Ex. I need to train the model before apply scoring or visualization,
Visualization is done on the training set and not the scoring since the dataset
of the scoring will have no information about True positive True Negative)
In order to extract data from snowflake it is necessary to decide if we want to download 
the train the scoring or both datasets and this will be set in the config.yml:
```
sql_extract_train = True
sql_extract_scoring = True
train = True
scoring = True
```

if you would like just to run the training process and apply the visualization step then:
```
sql_extract_train = True
train = True
scoring = False
visual= True
```

or if your model have been already trained and you would like to apply the scores then:
```
sql_extract_scoring = True
train = False
scoring = True
```
In addition, if you have been already downloading the data and you would like not to repeat this
step than :
```
sql_extract_train = False
train = True
scoring = False
```
or
```
sql_extract_scoring = False
train = False
scoring = True
```

During the training process the data set is split into (train, test and eval)
and in addition a weight is used in the training to unbalance the train set.

There are two ways of splitting the data set:
1. random split:
- train 60%
- test 20%
- eval 20%

```
random_split: True
ratio_weight: # imbalance ratio
```

2. time dependent split: 
- test = data set with 'full_dt' >= last_date_test,
- eval = 25 % of remaining data set,
- train = 75 % of remaining data set.

```
random_split: False
ratio_weight: # imbalance ratio
last_date_test: '2018-01-31'
```

## Feature extraction
All the sql statements have to start with a space at the beginning.

## Model Pipeline

This Pipeline has been build in order to accept a balanced train data set. Here we will
try to get an idea on what is happening during the training step.

1. From the input train data set(from sql extraction) we will filter just the features 
contained in the file  ' abt_vars.txt '

2. From the types  in the input train data set the features are divided into categorical 
and numerical

3. The input train data set is divided in three disjoint sets: Train, Eval, Test

4. The Sklearn Pipeline is created:

- Pre processing step: feature selection, imputation and one hot encoding
- model: xgboost

5. The Sklearn Pipeline is trained using cross validation.
The Train set is used to train the model using the given weights and the Eval
set to measure the performance in order to find the best parameters set.

6. The final model is trained on train set with the best parameters obtained from the
cross validation

7. Calculate the auc of the test set with the final trained model

8. The trained model, the test set, the statistics on the features are stored

## Visualization

Once the model have been trained and the test set have been stored,
it is possible to run the visualization that will provide confusion matrix, auc, Shaply values
for the final model.

## Result folder
All data and files are stored in a result folder created in the path given as input or defined in the "main_dir" key
from config.yml.

At every run the folders of docs gets updated with the reports and the new plots.
