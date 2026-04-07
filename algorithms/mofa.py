'''
Multiobjective firefly optimizer

@author Aidan Ryther arr9180
'''

import numpy as np


"""
dominates(g_i, h_i, g_j, h_j): g_i and h_i are the two objective
values for solution i. g_j and h_j are the two objective values for
solution j. This checks if i Pareto-dominates j for minimization.
"""
def dominates(g_i, h_i, g_j, h_j):
    no_worse = g_i <= g_j and h_i <= h_j
    strictly_better = g_i < g_j or h_i < h_j
    return no_worse and strictly_better


"""
non_dominated_mask(g_X, h_X): g_X and h_X are the two objective
vectors for the current population. This returns the mask for the
non-dominated solutions.
"""
def non_dominated_mask(g_X, h_X):
    g_vals = np.asarray(g_X, dtype=float).reshape(-1)
    h_vals = np.asarray(h_X, dtype=float).reshape(-1)
    n_pop = np.shape(g_vals)[0]
    keep = np.ones(n_pop, dtype=bool)

    for i in range(n_pop):
        for j in range(n_pop):
            if i == j:
                continue

            if dominates(g_vals[j], h_vals[j], g_vals[i], h_vals[i]):
                keep[i] = False
                break

    return keep


"""
random_weights(n_obj): n_obj is the number of objectives. This makes
random weights that add up to 1.
"""
def random_weights(n_obj):
    weights = np.random.random(n_obj)
    weight_sum = np.sum(weights)
    return weights / weight_sum


"""
multiobjective_firefly(X, f, g, h, n_epoch, beta0, gamma, alpha0):
X is the starting firefly matrix. f is the scalar project cost used
to pick one final best solution. g and h are the two Pareto
objectives. n_epoch is the number of iterations. beta0, gamma, and
alpha0 are the firefly settings. This runs the multiobjective
firefly algorithm.
"""
def multiobjective_firefly(X, f, g, h, n_epoch, beta0, gamma, alpha0):
    if g is None or h is None:
        raise ValueError("multiobjective firefly needs two objective functions")

    X = np.asarray(X, dtype=float).copy()

    if X.ndim != 2:
        raise ValueError("X must be a 2D matrix")

    n_pop, n_dim = np.shape(X)
    alpha = alpha0

    g_X = np.asarray(g(X), dtype=float).reshape(-1)
    h_X = np.asarray(h(X), dtype=float).reshape(-1)

    if np.shape(g_X)[0] != n_pop or np.shape(h_X)[0] != n_pop:
        raise ValueError("objective functions must return one value per solution")

    keep = non_dominated_mask(g_X, h_X)
    X_front = X[keep].copy()
    g_front = g_X[keep].copy()
    h_front = h_X[keep].copy()

    for t in range(n_epoch):
        g_X = np.asarray(g(X), dtype=float).reshape(-1)
        h_X = np.asarray(h(X), dtype=float).reshape(-1)
        keep = non_dominated_mask(g_X, h_X)

        X_all = np.vstack((X_front, X))
        g_all = np.concatenate((g_front, g_X))
        h_all = np.concatenate((h_front, h_X))

        weights = random_weights(2)
        weighted_cost = weights[0] * g_all + weights[1] * h_all
        best_idx = np.argmin(weighted_cost)
        x_weighted_best = X_all[best_idx].copy()

        X_new = X.copy()

        for i in range(n_pop):
            if keep[i]:
                noise = alpha * np.random.normal(0.0, 1.0, n_dim)
                X_new[i] = x_weighted_best + noise
                continue

            for j in range(n_pop):
                if i == j:
                    continue

                if dominates(g_X[j], h_X[j], g_X[i], h_X[i]):
                    dist_sq = np.sum((X_new[i] - X[j]) ** 2)
                    beta = beta0 * np.exp(-gamma * dist_sq)
                    noise = alpha * np.random.normal(0.0, 1.0, n_dim)
                    X_new[i] = X_new[i] + beta * (X[j] - X_new[i]) + noise

        X = X_new
        g_X = np.asarray(g(X), dtype=float).reshape(-1)
        h_X = np.asarray(h(X), dtype=float).reshape(-1)

        X_all = np.vstack((X_front, X))
        g_all = np.concatenate((g_front, g_X))
        h_all = np.concatenate((h_front, h_X))

        keep = non_dominated_mask(g_all, h_all)
        X_front = X_all[keep].copy()
        g_front = g_all[keep].copy()
        h_front = h_all[keep].copy()

        alpha = alpha0 * (0.9 ** (t + 1))

    f_front = np.asarray(f(X_front), dtype=float)
    if f_front.ndim == 2 and np.shape(f_front)[1] == 1:
        f_front = f_front[:, 0]
    else:
        f_front = np.reshape(f_front, (-1,))

    best_idx = np.argmin(f_front)
    g_star = X_front[best_idx].copy()
    f_g_star = f_front[best_idx]

    return g_star, f_g_star, X_front, f_front
