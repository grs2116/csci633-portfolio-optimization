'''
Genetic algorithm optimizer

@author Tiffany Lee tl1105
'''
import numpy as np


def cost_values(f, X):
    """
    Helper function that evaluates the cost function 
    and ensures it returns a 1D array of costs.

    :param f: cost function to evaluate
    :param X: population of solutions to evaluate

    :return: 1D array of cost values corresponding to each solution in X
    """
    f_X = np.asarray(f(X), dtype=float)
    if f_X.ndim == 2 and np.shape(f_X)[1] == 1:
        return f_X[:, 0]
    return np.reshape(f_X, (-1,))


def normalize_population(X):
    """
    Helper function that normalizes the population so that each solution
    is a valid probability distribution (non-negative and sums to 1).
    
    :param X: population of solutions to normalize
    
    :return: normalized population where each solution is non-negative 
    and sums to 1
    """
    X = np.maximum(np.asarray(X, dtype=float), 0.0)
    row_sum = np.sum(X, axis=1, keepdims=True)

    zero_rows = row_sum[:, 0] <= 1e-12
    if np.any(zero_rows):
        X[zero_rows] = 1.0 / np.shape(X)[1]
        row_sum = np.sum(X, axis=1, keepdims=True)

    return X / row_sum


def select_parent(X, f_X, tournament_size):
    """
    Helper function that selects a parent solution from 
    the population using tournament selection.

    :param X: population of solutions to select from
    :param f_X: cost values corresponding to each solution in X
    :param tournament_size: number of solutions to randomly select for the tournament

    :return: selected parent solution from the tournament
    """
    n_pop = np.shape(X)[0]
    tournament_size = max(2, min(tournament_size, n_pop))

    if tournament_size == n_pop:
        idx = np.arange(n_pop)
    else:
        idx = np.random.choice(n_pop, size=tournament_size, replace=False)

    winner_idx = idx[np.argmin(f_X[idx])]
    return X[winner_idx].copy()



"""
genetic_algorithm(X, f, g, h, n_epoch, mutation_rate,
crossover_rate): X is the starting population. f is the cost
function. g and h are extra inputs kept for consistency. n_epoch
is the number of generations. mutation_rate and crossover_rate are
the GA settings. This will run the genetic algorithm.
"""
def genetic_algorithm(X, f, g, h, n_epoch, mutation_rate, crossover_rate):
    """
    Genetic algorithm optimizer, fucntion that evolves a population of 
    solutions to minimize a cost function.

    :param X: starting population of solutions (2D array -> row solution, col is gene)
    :param f: cost function to minimize
    :param g: extra input for consistency (not used in this algorithm)
    :param h: extra input for consistency (not used in this algorithm)
    :param n_epoch: number of generations to evolve the population
    :param mutation_rate: probability of mutating each gene in the child solution
    :param crossover_rate: probability of performing crossover between two parents

    :return: tuple containing the best solution found, its cost, 
    the final population, and their costs
    """
    X = X.copy()

    if np.shape(X)[0] < 2:
        raise ValueError("Genetic Algorithm needs at least 2 solutions")

    X = normalize_population(X)
    f_X = cost_values(f, X)

    g_idx = np.argmin(f_X)
    g_star = X[g_idx].copy()
    f_g_star = f_X[g_idx]

    elite_frac = 0.10
    elite_count = max(1, int(np.floor(elite_frac * np.shape(X)[0])))
    mutation_scale = 0.05
    tournament_size = 3

    for _ in range(n_epoch):
        ranked_idx = np.argsort(f_X)
        elites = X[ranked_idx[:elite_count]].copy()

        X_new = elites.copy()
        n_pop, n_dim = np.shape(X)

        while np.shape(X_new)[0] < n_pop:
            parent_a = select_parent(X, f_X, tournament_size)
            parent_b = select_parent(X, f_X, tournament_size)

            child = parent_a.copy()

            if np.random.random() <= crossover_rate:
                alpha = np.random.random(n_dim)
                child = alpha * parent_a + (1.0 - alpha) * parent_b

            mutation_mask = np.random.random(n_dim) < mutation_rate
            if np.any(mutation_mask):
                child[mutation_mask] += np.random.normal(0.0, mutation_scale, np.sum(mutation_mask))

            X_new = np.vstack((X_new, child.reshape(1, -1)))

        X = normalize_population(X_new[:n_pop])
        f_X = cost_values(f, X)

        g_idx = np.argmin(f_X)
        if f_X[g_idx] < f_g_star:
            g_star = X[g_idx].copy()
            f_g_star = f_X[g_idx]

    return g_star, f_g_star, X, f_X