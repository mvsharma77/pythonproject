#!/usr/bin/python

import os
import subprocess

import pkg_resources

# variable
sys_pkg = []
sys_pkg_approx = []
sys_pkg_keys = []
req_pkg_list = []
missing = []
sys_pkg_dict = {}
temp_place_holder = []


# To get the packages installed on users machine
# get_metadata('-')Return the named metadata resource as a string, and '-' is a positional mandatory input.
# The data is read in binary mode; i.e., the exact bytes of the resource file are returned.
def get_sys_packages():
    packages = pkg_resources.working_set
    packages_dict = {}
    for package in packages:
        modules_from_package = package._get_metadata('-')
        for module in modules_from_package:
            packages_dict[module] = package.version
        packages_dict[package.key] = package.version
        packages_dict[package.project_name] = package.version
    return packages_dict


def get_req_packages():
    # Provide file path to access requirement.txt file
    # req_file_path = os.path.dirname(os.path.realpath("requirements.txt"))
    req_file_path = open(r'requirements.txt')
    req_module = req_file_path.readlines()
    # Removing new line character /n, and getting data in list format
    req_module = [module.strip() for module in req_module]
    return req_module


def get_missing_packages():
    sys_pkg_dict = get_sys_packages()
    # List of required packages
    req_pkg_list = str(get_req_packages()).replace(' ', '').replace('[', '').replace(']', '').split(',')
    # Converting the dictionary output into formatted list
    sys_pkg = [('{}=={}'.format(key, value)) for (key, value) in sys_pkg_dict.items()]
    sys_pkg_approx = [x[:-2] for x in sys_pkg]  # For comparison with approximate compatible version
    sys_pkg_keys = [*sys_pkg_dict.keys()]  # For latest version
    temp_place_holder = list(map(chk_pkg_name_versions, req_pkg_list))
    return temp_place_holder


def chk_pkg_name_versions(req_pkg_list):
    item = req_pkg_list
    if '==' in item:
        if str(item).replace("'", '') not in sys_pkg:
            missing.append(str(item).replace("'", ''))
    elif '~=' in item:
        if str(item)[0:len(str(item)) - 3].replace("'", '').replace('~', '=') not in sys_pkg_approx:
            missing.append(str(item).replace("'", ''))
    else:
        if str(item).replace("'", '') not in sys_pkg_keys and str(item).replace("'", '')[0] != '-':
            missing.append(str(item).replace("'", ''))
    return item


def load_packages():
    get_missing_packages()
    pkg_to_be_installed = missing
    # Preparation of command
    cmd = str('python -m pip install ') + str(pkg_to_be_installed).replace("'",'').replace('[','').replace(']','').replace(',','')
    # Execution
    if missing:
        subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL)
    return


load_packages()
