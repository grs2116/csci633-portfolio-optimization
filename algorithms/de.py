'''
Differential evolution optimizer

@author Aidan Ryther arr9180
'''

import numpy as np


"""
differential_evolution(X, f, g, h, n_epoch, diff_weight,
crossover_prob): X is the starting population. f is the cost
function. g and h are extra inputs kept for consistency. n_epoch
is the number of generations. diff_weight is the DE mutation
weight. crossover_prob is the DE crossover setting. This runs
differential evolution.
"""
def differential_evolution(X, f, g, h, n_epoch, diff_weight, crossover_prob):
    X = X.copy()
    
    if np.shape(X)[0] < 4:
        raise ValueError("differential evolution needs at least 4 solutions")
    
    f_X = f(X)

    g_idx = np.argmin(f_X)
    g_star = X[g_idx].copy()
    f_g_star = f_X[g_idx]

    for _ in range(n_epoch):
        X_new = X.copy()
        n_pop, n_dim = np.shape(X)
        idx_vals = np.arange(n_pop)

        for i in range(n_pop):
            other_idx = idx_vals[idx_vals != i]
            p_idx, q_idx, r_idx = np.random.choice(other_idx, 3, replace=False)

            x_p = X[p_idx]
            x_q = X[q_idx]
            x_r = X[r_idx]

            donor = x_p + diff_weight * (x_q - x_r)
            trial = X[i].copy()

            force_idx = np.random.randint(n_dim)

            for j in range(n_dim):
                if np.random.random() <= crossover_prob or j == force_idx:
                    trial[j] = donor[j]

            X_new[i] = trial

        f_X_new = f(X_new)

        improved = f_X_new < f_X
        X[improved] = X_new[improved]
        f_X[improved] = f_X_new[improved]

        g_idx = np.argmin(f_X)
        g_star = X[g_idx].copy()
        f_g_star = f_X[g_idx]

    return g_star, f_g_star, X, f_X
