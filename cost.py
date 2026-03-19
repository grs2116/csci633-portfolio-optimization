'''
Portfolio cost functions

@author Aidan Ryther arr9180
'''

"""
normalize_weights(weights): weights is a raw portfolio weight vector
or matrix. This turns it into valid portfolio weights.
"""
def normalize_weights(weights):
    pass


"""
portfolio_return(weights, mean_returns): weights is the portfolio
weight input. mean_returns is the vector of asset mean returns.
This computes the portfolio return.
"""
def portfolio_return(weights, mean_returns):
    pass


"""
portfolio_risk(weights, covariance_matrix): weights is the
portfolio weight input. covariance_matrix is the asset covariance
matrix. This computes the portfolio risk.
"""
def portfolio_risk(weights, covariance_matrix):
    pass


"""
portfolio_cost(weights, mean_returns, covariance_matrix,
risk_weight, max_weight=None): weights is the portfolio weight
input. mean_returns is the vector of asset mean returns.
covariance_matrix is the asset covariance matrix. risk_weight is
the return versus risk tradeoff. max_weight is the optional cap on
one asset. This computes the one cost function all algorithms use.
"""
def portfolio_cost(weights, mean_returns, covariance_matrix, risk_weight, max_weight=None):
    pass
