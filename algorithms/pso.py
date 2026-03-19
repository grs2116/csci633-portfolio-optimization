'''
Particle swarm optimizer

@author Grayson Siegler grs2116
'''

import numpy as np

"""
update_velocities(X, velocities, alpha, beta, theta, g_star,
X_star): X is the current particle matrix. velocities is the
current velocity matrix. alpha, beta, and theta are the PSO
settings. g_star is the global best solution. X_star is the local
best matrix. This updates the particle velocities.
"""
def update_velocities(X, velocities, alpha, beta, theta, g_star, X_star):
    eps1 = np.random.random(np.shape(velocities))
    eps2 = np.random.random(np.shape(velocities))
    return theta * velocities + alpha * eps1 * (g_star - X) + beta * eps2 * (X_star - X)

"""
particle_swarm(X, velocities, f, g, h, n_epoch, alpha, beta,
theta): X is the starting particle matrix. velocities is the
starting velocity matrix. f is the cost function. g and h are
extra inputs kept for consistency. n_epoch is the number of swarm
steps. alpha, beta, and theta are the PSO settings. This runs
particle swarm optimization.
"""
def particle_swarm(X, velocities, f, g, h, n_epoch, alpha, beta, theta):
    X = X.copy()
    velocities = velocities.copy()

    X_star = X.copy()
    f_X_star = f(X_star)

    g_idx = np.argmin(f_X_star)
    g_star = X_star[g_idx].copy()
    f_g_star = f_X_star[g_idx]

    for _ in range(n_epoch):
        velocities = update_velocities(X, velocities, alpha, beta, theta, g_star, X_star)
        X = X + velocities

        f_X = f(X)  # N
        improved = f_X < f_X_star

        X_star[improved] = X[improved]
        f_X_star[improved] = f_X[improved]

        g_idx = np.argmin(f_X_star)
        g_star = X_star[g_idx].copy()
        f_g_star = f_X_star[g_idx]

    return g_star, f_g_star, X_star, f_X_star
