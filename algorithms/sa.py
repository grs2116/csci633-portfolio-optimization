'''
Simulated annealing optimizer

@author Aidan Ryther arr9180
'''

import numpy as np


"""
simulated_annealing(X, f, g, h, n_epoch, step_size, temp_init,
temp_decay): X is the starting solution matrix. f is the cost
function. g and h are extra inputs kept for consistency. n_epoch
is the number of annealing steps. step_size is the move size.
temp_init is the starting temperature. temp_decay is the cooling
value. This runs simulated annealing.
"""
def simulated_annealing(X, f, g, h, n_epoch, step_size, temp_init, temp_decay):
    X = X.copy()

    X_star = X.copy()
    f_X = f(X)
    f_X_star = f_X.copy()

    g_idx = np.argmin(f_X_star)
    g_star = X_star[g_idx].copy()
    f_g_star = f_X_star[g_idx]

    temp = temp_init

    for _ in range(n_epoch):
        noise = np.random.normal(0.0, step_size, np.shape(X))
        X_new = X + noise
        f_X_new = f(X_new)

        safe_temp = max(temp, 1e-12)
        delta = f_X_new - f_X
        accept = (delta <= 0) | (np.random.random(np.shape(f_X)) < np.exp(-delta / safe_temp))

        X[accept] = X_new[accept]
        f_X[accept] = f_X_new[accept]

        improved = f_X < f_X_star
        X_star[improved] = X[improved]
        f_X_star[improved] = f_X[improved]

        g_idx = np.argmin(f_X_star)
        g_star = X_star[g_idx].copy()
        f_g_star = f_X_star[g_idx]

        temp = temp * temp_decay

    return g_star, f_g_star, X_star, f_X_star
