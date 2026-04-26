'''
Genetic algorithm optimizer

@author Tiffany Lee tl1105
'''
import numpy as np


"""
cost_values(f, X): f is the cost function to evaluate and X is the
population of solutions to evaluate. This helper evaluates the cost
function and returns a 1D vector of costs.
"""
def cost_values(f, X):
    f_X = np.asarray(f(X), dtype=float)
    if f_X.ndim == 2 and np.shape(f_X)[1] == 1:
        return f_X[:, 0]
    return np.reshape(f_X, (-1,))


"""
normalize_population(X): X is the population to normalize. This helper
makes each solution a valid probability distribution by enforcing
non-negativity and row sums of 1.
"""
def normalize_population(X):
    X = np.maximum(np.asarray(X, dtype=float), 0.0)
    row_sum = np.sum(X, axis=1, keepdims=True)

    zero_rows = row_sum[:, 0] <= 1e-12
    if np.any(zero_rows):
        X[zero_rows] = 1.0 / np.shape(X)[1]
        row_sum = np.sum(X, axis=1, keepdims=True)

    return X / row_sum


"""
select_parent(X, f_X, tournament_size): X is the population and f_X is
the matching cost vector. tournament_size is the number of candidates
for tournament selection. This helper returns one selected parent.
"""
def select_parent(X, f_X, tournament_size):
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