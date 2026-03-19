'''
Portfolio evaluation helpers

@author Aidan Ryther arr9180
'''

"""
evaluate_solution(weights, mean_returns, covariance_matrix):
weights is the portfolio weight input. mean_returns is the vector
of asset mean returns. covariance_matrix is the asset covariance
matrix. This scores one portfolio solution.
"""
def evaluate_solution(weights, mean_returns, covariance_matrix):
    pass


"""
run_one_trial(algorithm, algorithm_args, cost_fn): algorithm is the
algorithm function to run. algorithm_args holds that algorithm's
inputs. cost_fn is the portfolio cost function. This runs one
trial.
"""
def run_one_trial(algorithm, algorithm_args, cost_fn):
    pass


"""
run_trials(algorithm, algorithm_args, cost_fn, n_trials):
algorithm is the algorithm function to run. algorithm_args holds
that algorithm's inputs. cost_fn is the portfolio cost function.
n_trials is the number of runs. This runs many trials.
"""
def run_trials(algorithm, algorithm_args, cost_fn, n_trials):
    pass


"""
summarize_results(results): results is the collection of trial
outputs. This computes the final summary values.
"""
def summarize_results(results):
    pass


"""
plot_results(results): results is the collection of final
experiment outputs. This makes the final plots.
"""
def plot_results(results):
    pass
