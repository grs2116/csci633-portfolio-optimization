'''
Portfolio data loader

@authors
Aidan Ryther arr9180
Tiffany Lee tl1105
Grayson Siegler grs2116
'''

import os

import numpy as np


"""
load_portfolio_file(file_name): file_name is one portfolio data
file. This loads one dataset and returns its basic values, the
correlation matrix, and the covariance matrix.
"""
def load_portfolio_file(file_name):
    with open(file_name, "r") as inFile:
        raw_lines = inFile.readlines()

    lines = [line.strip() for line in raw_lines if line.strip() != ""]

    n_assets = int(lines[0])

    mean_returns = []
    std_devs = []

    for i in range(n_assets):
        curr_line = lines[i + 1].split()
        mean_val = float(curr_line[0])
        std_val = float(curr_line[1])

        mean_returns.append(mean_val)
        std_devs.append(std_val)

    mean_returns = np.array(mean_returns, dtype=float)
    std_devs = np.array(std_devs, dtype=float)

    corr_lines = lines[1 + n_assets:]
    expected_corr = (n_assets * (n_assets + 1)) // 2

    if len(corr_lines) != expected_corr:
        raise ValueError("wrong number of correlation lines in portfolio file")

    correlations = np.eye(n_assets, dtype=float)

    for line in corr_lines:
        curr_line = line.split()
        i_idx = int(curr_line[0]) - 1
        j_idx = int(curr_line[1]) - 1
        corr_val = float(curr_line[2])

        correlations[i_idx, j_idx] = corr_val
        correlations[j_idx, i_idx] = corr_val

    covariance_matrix = build_covariance_matrix(std_devs, correlations)

    name = os.path.basename(file_name)
    name = os.path.splitext(name)[0]

    data = {
        "name": name,
        "n_assets": n_assets,
        "mean_returns": mean_returns,
        "std_devs": std_devs,
        "correlations": correlations,
        "covariance_matrix": covariance_matrix,
    }
    return data


"""
build_covariance_matrix(std_devs, correlations): std_devs is the
vector of asset standard deviations. correlations is the asset
correlation matrix. This builds and returns the covariance matrix.
"""
def build_covariance_matrix(std_devs, correlations):
    std_vals = np.asarray(std_devs, dtype=float)
    corr_vals = np.asarray(correlations, dtype=float)

    std_outer = np.outer(std_vals, std_vals)
    covariance_matrix = corr_vals * std_outer
    return covariance_matrix


"""
load_all_datasets(data_dir): data_dir is the folder holding the
portfolio files. This loads port1 through port5 and returns them in
one dictionary.
"""
def load_all_datasets(data_dir):
    file_names = [
        "port1.txt",
        "port2.txt",
        "port3.txt",
        "port4.txt",
        "port5.txt",
    ]

    datasets = {}

    for file_name in file_names:
        curr_path = os.path.join(data_dir, file_name)

        if not os.path.exists(curr_path):
            raise FileNotFoundError(curr_path)

        curr_data = load_portfolio_file(curr_path)
        datasets[curr_data["name"]] = curr_data

    return datasets
