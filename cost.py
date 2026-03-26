'''
Portfolio cost functions

@authors
Aidan Ryther arr9180
Tiffany Lee tl1105
Grayson Siegler grs2116
'''

import numpy as np


"""
normalize_weights(weights, max_weight=None): weights is a raw
portfolio weight vector or matrix. max_weight is the optional largest
allowed weight for one asset. This turns the input into valid
portfolio weights.
"""
def normalize_weights(weights, max_weight=None):
    weight_vals = np.asarray(weights, dtype=float)
    vector_input = False

    # Keep the input row-based like the other files.
    if weight_vals.ndim == 1:
        weight_vals = weight_vals.reshape(1, -1)
        vector_input = True
    elif weight_vals.ndim != 2:
        raise ValueError("weights must be a vector or matrix")

    weight_vals = weight_vals.copy()

    # Drop negative weights.
    weight_vals = np.maximum(weight_vals, 0.0)
    row_sum = np.sum(weight_vals, axis=1, keepdims=True)

    # If a row is all zero, use equal weights.
    zero_rows = row_sum[:, 0] <= 1e-12
    if np.any(zero_rows):
        n_assets = np.shape(weight_vals)[1]
        weight_vals[zero_rows] = 1.0 / n_assets
        row_sum = np.sum(weight_vals, axis=1, keepdims=True)

    # Make each row sum to 1.
    weight_vals = weight_vals / row_sum

    # Apply a max cap if one is used.
    if max_weight is not None:
        if max_weight <= 0.0 or max_weight > 1.0:
            raise ValueError("max_weight must be in (0, 1]")

        n_assets = np.shape(weight_vals)[1]
        if max_weight * n_assets < 1.0 - 1e-12:
            raise ValueError("max_weight is too small for a fully invested portfolio")

        tol = 1e-12

        for i in range(np.shape(weight_vals)[0]):
            curr_weights = weight_vals[i].copy()

            while True:
                over_limit = curr_weights > max_weight + tol
                if not np.any(over_limit):
                    break

                curr_weights[over_limit] = max_weight

                under_limit = curr_weights < max_weight - tol
                fixed_weight = np.sum(curr_weights[~under_limit])
                remaining_weight = 1.0 - fixed_weight

                if remaining_weight <= tol:
                    curr_weights[under_limit] = 0.0
                    break

                if not np.any(under_limit):
                    break

                under_sum = np.sum(curr_weights[under_limit])

                if under_sum <= tol:
                    curr_weights[under_limit] = remaining_weight / np.sum(under_limit)
                else:
                    curr_weights[under_limit] = curr_weights[under_limit] / under_sum
                    curr_weights[under_limit] = curr_weights[under_limit] * remaining_weight

            curr_sum = np.sum(curr_weights)
            if curr_sum <= tol:
                curr_weights[:] = 1.0 / n_assets
            else:
                curr_weights = curr_weights / curr_sum

            weight_vals[i] = curr_weights

    if vector_input:
        return weight_vals[0]
    return weight_vals


"""
portfolio_return(weights, mean_returns): weights is the portfolio
weight input. mean_returns is the vector of asset mean returns.
This computes the portfolio return.
"""
def portfolio_return(weights, mean_returns):
    weight_vals = normalize_weights(weights)
    mean_vals = np.asarray(mean_returns, dtype=float).reshape(-1)
    vector_input = False

    if np.ndim(weight_vals) == 1:
        weight_vals = weight_vals.reshape(1, -1)
        vector_input = True

    if np.shape(weight_vals)[1] != np.shape(mean_vals)[0]:
        raise ValueError("weights and mean_returns have different sizes")

    # Weighted sum of the mean returns.
    mean_row = mean_vals.reshape(1, -1)
    weighted_returns = weight_vals * mean_row
    return_vals = np.sum(weighted_returns, axis=1)

    if vector_input:
        return float(return_vals[0])
    return return_vals


"""
portfolio_risk(weights, covariance_matrix): weights is the
portfolio weight input. covariance_matrix is the asset covariance
matrix. This computes the portfolio risk.
"""
def portfolio_risk(weights, covariance_matrix):
    weight_vals = normalize_weights(weights)
    cov_vals = np.asarray(covariance_matrix, dtype=float)
    vector_input = False

    if np.ndim(weight_vals) == 1:
        weight_vals = weight_vals.reshape(1, -1)
        vector_input = True

    if cov_vals.ndim != 2 or np.shape(cov_vals)[0] != np.shape(cov_vals)[1]:
        raise ValueError("covariance_matrix must be square")

    if np.shape(weight_vals)[1] != np.shape(cov_vals)[0]:
        raise ValueError("weights and covariance_matrix have different sizes")

    # This is w^T * Sigma * w, row by row.
    cov_weighted = weight_vals @ cov_vals
    row_products = cov_weighted * weight_vals
    risk_vals = np.sum(row_products, axis=1)
    risk_vals = np.maximum(risk_vals, 0.0)

    if vector_input:
        return float(risk_vals[0])
    return risk_vals


"""
portfolio_cost(weights, mean_returns, covariance_matrix,
risk_weight, max_weight=None): weights is the portfolio weight
input. mean_returns is the vector of asset mean returns.
covariance_matrix is the asset covariance matrix. risk_weight is
the return versus risk tradeoff. max_weight is the optional cap on
one asset. This computes the one cost function all algorithms use.
"""
def portfolio_cost(weights, mean_returns, covariance_matrix, risk_weight, max_weight=None):
    if risk_weight < 0.0:
        raise ValueError("risk_weight must be non-negative")

    # Fix the raw search vectors first.
    weight_vals = normalize_weights(weights, max_weight=max_weight)

    # Get return and risk from the repaired weights.
    return_vals = portfolio_return(weight_vals, mean_returns)
    risk_vals = portfolio_risk(weight_vals, covariance_matrix)

    # Higher return lowers cost. Higher risk raises it.
    cost_vals = risk_weight * risk_vals - return_vals

    if np.ndim(weight_vals) == 1:
        return float(cost_vals)
    return cost_vals


"""
Cost model summary:

The portfolio objective follows the standard mean-variance form:

    return(w) = w^T * mu
    risk(w) = w^T * Sigma * w
    cost(w) = risk_weight * risk(w) - return(w)

where:
    w = portfolio weight vector
    mu = vector of asset mean returns
    Sigma = covariance matrix

Function roles:

normalize_weights(weights, max_weight=None)
    Fixes a raw portfolio vector.
    It removes negative weights, makes the weights add up to 1,
    and can enforce a max weight for each asset.

portfolio_return(weights, mean_returns)
    Gets expected portfolio return.
    It applies the portfolio weights to the asset mean returns and
    adds the results together.

portfolio_risk(weights, covariance_matrix)
    Gets portfolio variance.
    It uses the covariance matrix to measure the total risk of the
    weighted portfolio.

portfolio_cost(weights, mean_returns, covariance_matrix, risk_weight, max_weight=None)
    Main cost function.
    It fixes the weights, computes return and risk, and then combines
    them into one score.
    Higher return lowers the score.
    Higher risk raises the score.
    Lower cost is better.
"""
