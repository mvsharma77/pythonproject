#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Setup file to install automated scripts in your environment
All packages will be installed to python.site-packages
simply run:
    >>> python setup.py install
For a local installation or if you like to develop further
    >>> python setup.py develop --user
"""
import io
import os
import re
import sys
from setuptools import setup, find_packages
from src.classification import install_module

source_path = 'src'
path = os.path.abspath(os.path.dirname(sys.argv[0]))
os.chdir(path)
print(path)
packages = find_packages(source_path)
print(packages)


def read(filename):
    filename = os.path.join(os.path.dirname(__file__), filename)
    text_type = type(u"")
    with io.open(filename, mode="r", encoding='utf-8') as fd:
        return re.sub(text_type(r':[a-z]+:`~?(.*?)`'), text_type(r'``\1``'), fd.read())


def get_version_git():
    # The full version, including alpha/beta/rc tags.
    release = re.sub('^v', '', os.popen('git describe').read().strip())
    # The short X.Y version.
    return release


setup(
    name="classification",
    version=get_version_git(),
    url="",
    license='MIT',

    description="Classification model",
    keywords='Classification utils',
    long_description=read("README.md"),

    package_dir={'': source_path},
    packages=packages,
    zip_safe=False,
    package_data={'classification.configs': ['*.txt', '*.sql', '*.yaml', '*.csv']},

    include_package_data=True,

    entry_points={
        'console_scripts': [
            'ml_run_pipeline=classification.run_pipeline:main',
            'ml_show_config=classification.main_config:_Config.get_config',
        ]
    },

    #  install_requires=[
    #     "python-dateutil==2.8.0",
    #     "PyYAML",
    #     "matplotlib",
    #     "pandas",
    #     "pytest",
    #     "seaborn",
    #     "shap",
    #     "xgboost",
    #     "sphinx",
    #     "dill",
    #     "ruamel.yaml"
    #   ],

    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.7',
    ],
)

install_module.load_packages()
