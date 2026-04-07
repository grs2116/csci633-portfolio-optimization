'''
Portfolio evaluation helpers

@authors
Aidan Ryther arr9180
Tiffany Lee tl1105
Grayson Siegler grs2116
'''

import os
import inspect
import time

import numpy as np

from cost import normalize_weights
from cost import portfolio_return
from cost import portfolio_risk


"""
evaluate_solution(weights, mean_returns, covariance_matrix):
weights is the portfolio weight input. mean_returns is the vector
of asset mean returns. covariance_matrix is the asset covariance
matrix. This scores one portfolio solution.
"""
def evaluate_solution(weights, mean_returns, covariance_matrix):
    weight_vals = normalize_weights(weights)

    # Keep one solution as one vector.
    if np.ndim(weight_vals) == 2:
        if np.shape(weight_vals)[0] != 1:
            raise ValueError("evaluate_solution expects one portfolio solution")
        weight_vals = weight_vals[0]

    return_val = portfolio_return(weight_vals, mean_returns)
    risk_val = portfolio_risk(weight_vals, covariance_matrix)
    volatility_val = np.sqrt(max(risk_val, 0.0))

    safe_volatility = max(volatility_val, 1e-12)
    sharpe_like_val = return_val / safe_volatility

    result = {
        "weights": weight_vals,
        "return": float(return_val),
        "risk": float(risk_val),
        "volatility": float(volatility_val),
        "sharpe_like": float(sharpe_like_val),
    }
    return result


"""
run_one_trial(algorithm, algorithm_args, cost_fn): algorithm is the
algorithm function to run. algorithm_args holds that algorithm's
inputs. cost_fn is the portfolio cost function. This runs one
trial.
"""
def run_one_trial(algorithm, algorithm_args, cost_fn):
    if not isinstance(algorithm_args, dict):
        raise ValueError("algorithm_args must be a dictionary")

    curr_args = algorithm_args.copy()

    mean_returns = curr_args.pop("mean_returns", None)
    covariance_matrix = curr_args.pop("covariance_matrix", None)
    max_weight = curr_args.pop("max_weight", None)
    seed = curr_args.pop("seed", None)
    curr_args.pop("seed_base", None)

    if seed is not None:
        np.random.seed(int(seed))

    if "g" not in curr_args:
        curr_args["g"] = None

    if "h" not in curr_args:
        curr_args["h"] = None

    # Build fresh trial inputs if any argument was given as a function.
    for arg_name in list(curr_args.keys()):
        curr_val = curr_args[arg_name]

        if callable(curr_val):
            try:
                signature = inspect.signature(curr_val)
            except (TypeError, ValueError):
                signature = None

            if signature is None:
                continue

            needs_input = False

            for param in signature.parameters.values():
                if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                    continue

                if param.default is inspect._empty:
                    needs_input = True
                    break

            if not needs_input:
                curr_args[arg_name] = curr_val()

    curr_args["f"] = cost_fn

    start_time = time.perf_counter()
    best_weights, best_cost, final_population, final_costs = algorithm(**curr_args)
    end_time = time.perf_counter()
    runtime_seconds = end_time - start_time

    if max_weight is None:
        fixed_weights = normalize_weights(best_weights)
    else:
        fixed_weights = normalize_weights(best_weights, max_weight=max_weight)

    result = {
        "best_weights": fixed_weights,
        "best_cost": float(best_cost),
        "final_population": final_population,
        "final_costs": np.asarray(final_costs, dtype=float),
        "runtime_seconds": float(runtime_seconds),
    }

    if mean_returns is not None and covariance_matrix is not None:
        metrics = evaluate_solution(fixed_weights, mean_returns, covariance_matrix)
        metrics.pop("weights", None)
        result.update(metrics)

    return result


"""
run_trials(algorithm, algorithm_args, cost_fn, n_trials):
algorithm is the algorithm function to run. algorithm_args holds
that algorithm's inputs. cost_fn is the portfolio cost function.
n_trials is the number of runs. This runs many trials.
"""
def run_trials(algorithm, algorithm_args, cost_fn, n_trials):
    if n_trials <= 0:
        raise ValueError("n_trials must be positive")

    if not isinstance(algorithm_args, dict):
        raise ValueError("algorithm_args must be a dictionary")

    results = []

    for i in range(n_trials):
        curr_args = algorithm_args.copy()

        if "seed_base" in curr_args:
            curr_args["seed"] = int(curr_args["seed_base"]) + i
        elif "seed" in curr_args:
            curr_args["seed"] = int(curr_args["seed"]) + i

        trial_result = run_one_trial(algorithm, curr_args, cost_fn)
        trial_result["trial"] = i + 1
        results.append(trial_result)

    return results


"""
summarize_results(results): results is the collection of trial
outputs. This computes the final summary values.
"""
def summarize_results(results):
    if len(results) == 0:
        raise ValueError("results is empty")

    cost_vals = np.array([result["best_cost"] for result in results], dtype=float)
    best_idx = int(np.argmin(cost_vals))

    if len(cost_vals) > 1:
        cost_std = float(np.std(cost_vals, ddof=1))
    else:
        cost_std = 0.0

    summary = {
        "n_trials": len(results),
        "cost_mean": float(np.mean(cost_vals)),
        "cost_std": cost_std,
        "cost_best": float(np.min(cost_vals)),
        "cost_worst": float(np.max(cost_vals)),
        "best_trial": int(results[best_idx].get("trial", best_idx + 1)),
        "best_weights": results[best_idx]["best_weights"],
        "trial_results": results,
    }

    metric_names = ["return", "risk", "volatility", "sharpe_like"]

    for metric_name in metric_names:
        if metric_name in results[0]:
            metric_vals = np.array([result[metric_name] for result in results], dtype=float)

            if len(metric_vals) > 1:
                metric_std = float(np.std(metric_vals, ddof=1))
            else:
                metric_std = 0.0

            summary[metric_name + "_mean"] = float(np.mean(metric_vals))
            summary[metric_name + "_std"] = metric_std
            summary[metric_name + "_best"] = float(metric_vals[best_idx])

    if "runtime_seconds" in results[0]:
        runtime_vals = np.array([result["runtime_seconds"] for result in results], dtype=float)

        if len(runtime_vals) > 1:
            runtime_std = float(np.std(runtime_vals, ddof=1))
        else:
            runtime_std = 0.0

        summary["runtime_mean"] = float(np.mean(runtime_vals))
        summary["runtime_std"] = runtime_std
        summary["runtime_min"] = float(np.min(runtime_vals))
        summary["runtime_max"] = float(np.max(runtime_vals))

    return summary


"""
plot_results(results): results is the collection of final
experiment outputs. This makes the final plots.
"""
def plot_results(results):
    import matplotlib.pyplot as plt

    if not isinstance(results, dict) or len(results) == 0:
        raise ValueError("results must be a non-empty dictionary")

    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    color_map = {
        "sa": "#c0392b",
        "de": "#2563eb",
        "pso": "#1f9d55",
        "ga": "#7c3aed",
        "mofa": "#f59e0b",
    }

    first_val = next(iter(results.values()))

    # Flat case: {label: summary}
    if isinstance(first_val, dict) and "cost_mean" in first_val:
        labels = list(results.keys())
        labels.sort(key=lambda label: results[label]["cost_mean"])

        cost_means = [results[label]["cost_mean"] for label in labels]
        cost_stds = [results[label]["cost_std"] for label in labels]

        colors = []
        for label in labels:
            colors.append(color_map.get(str(label).lower(), "#bfc5cc"))

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(labels, cost_means, yerr=cost_stds, capsize=4, color=colors, edgecolor="black")
        ax.set_title("Mean Cost by Algorithm")
        ax.set_ylabel("Mean Cost")
        ax.set_xlabel("Algorithm")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.set_axisbelow(True)
        fig.tight_layout()

        summary_lines = []
        summary_lines.append("Algorithm Results")
        summary_lines.append("")

        for label in labels:
            curr_summary = results[label]
            summary_lines.append(label)
            summary_lines.append("  trials: " + str(curr_summary["n_trials"]))
            summary_lines.append("  cost mean: " + str(curr_summary["cost_mean"]))
            summary_lines.append("  cost std: " + str(curr_summary["cost_std"]))
            summary_lines.append("  cost best: " + str(curr_summary["cost_best"]))
            summary_lines.append("  cost worst: " + str(curr_summary["cost_worst"]))

            if "runtime_mean" in curr_summary:
                summary_lines.append("  runtime mean: " + str(curr_summary["runtime_mean"]))
                summary_lines.append("  runtime std: " + str(curr_summary["runtime_std"]))

            if "return_mean" in curr_summary:
                summary_lines.append("  return mean: " + str(curr_summary["return_mean"]))
                summary_lines.append("  risk mean: " + str(curr_summary["risk_mean"]))
                summary_lines.append("  volatility mean: " + str(curr_summary["volatility_mean"]))
                summary_lines.append("  sharpe_like mean: " + str(curr_summary["sharpe_like_mean"]))

            summary_lines.append("")

        summary_path = os.path.join(results_dir, "summary.txt")
        with open(summary_path, "w") as outFile:
            outFile.write("\n".join(summary_lines))

        figure_path = os.path.join(results_dir, "mean_cost_by_algorithm.png")
        fig.savefig(figure_path, dpi=200)
        return fig, ax

    # Nested case: {dataset_name: {algorithm_name: summary}}
    dataset_names = list(results.keys())
    n_datasets = len(dataset_names)

    fig, axes = plt.subplots(n_datasets, 1, figsize=(8, 4 * n_datasets))

    if n_datasets == 1:
        axes = [axes]

    for i in range(n_datasets):
        dataset_name = dataset_names[i]
        dataset_results = results[dataset_name]

        algo_names = list(dataset_results.keys())
        algo_names.sort(key=lambda name: dataset_results[name]["cost_mean"])

        cost_means = [dataset_results[name]["cost_mean"] for name in algo_names]
        cost_stds = [dataset_results[name]["cost_std"] for name in algo_names]

        colors = []
        for algo_name in algo_names:
            colors.append(color_map.get(str(algo_name).lower(), "#bfc5cc"))

        curr_ax = axes[i]
        curr_ax.bar(algo_names, cost_means, yerr=cost_stds, capsize=4, color=colors, edgecolor="black")
        curr_ax.set_title(dataset_name + " Mean Cost")
        curr_ax.set_ylabel("Mean Cost")
        curr_ax.set_xlabel("Algorithm")
        curr_ax.grid(axis="y", linestyle="--", alpha=0.3)
        curr_ax.set_axisbelow(True)

    fig.tight_layout()

    summary_lines = []
    summary_lines.append("Dataset Results")
    summary_lines.append("")

    for dataset_name in dataset_names:
        dataset_results = results[dataset_name]
        algo_names = list(dataset_results.keys())
        algo_names.sort(key=lambda name: dataset_results[name]["cost_mean"])

        summary_lines.append(dataset_name)
        summary_lines.append("")

        for algo_name in algo_names:
            curr_summary = dataset_results[algo_name]
            summary_lines.append("  " + algo_name)
            summary_lines.append("    trials: " + str(curr_summary["n_trials"]))
            summary_lines.append("    cost mean: " + str(curr_summary["cost_mean"]))
            summary_lines.append("    cost std: " + str(curr_summary["cost_std"]))
            summary_lines.append("    cost best: " + str(curr_summary["cost_best"]))
            summary_lines.append("    cost worst: " + str(curr_summary["cost_worst"]))

            if "runtime_mean" in curr_summary:
                summary_lines.append("    runtime mean: " + str(curr_summary["runtime_mean"]))
                summary_lines.append("    runtime std: " + str(curr_summary["runtime_std"]))

            if "return_mean" in curr_summary:
                summary_lines.append("    return mean: " + str(curr_summary["return_mean"]))
                summary_lines.append("    risk mean: " + str(curr_summary["risk_mean"]))
                summary_lines.append("    volatility mean: " + str(curr_summary["volatility_mean"]))
                summary_lines.append("    sharpe_like mean: " + str(curr_summary["sharpe_like_mean"]))

            summary_lines.append("")

        summary_lines.append("")

    summary_path = os.path.join(results_dir, "summary.txt")
    with open(summary_path, "w") as outFile:
        outFile.write("\n".join(summary_lines))

    figure_path = os.path.join(results_dir, "mean_cost_by_dataset.png")
    fig.savefig(figure_path, dpi=200)
    return fig, axes


"""
Evaluate summary:

evaluate_solution(weights, mean_returns, covariance_matrix)
    Scores one final portfolio.
    It returns the cleaned weights, expected return, variance,
    volatility, and a simple return-to-volatility ratio.

run_one_trial(algorithm, algorithm_args, cost_fn)
    Runs one algorithm one time.
    It stores the best weights, best cost, final population, and
    final costs. If return and covariance data are given, it also
    stores the portfolio metrics for the best solution.

run_trials(algorithm, algorithm_args, cost_fn, n_trials)
    Runs the same algorithm many times.
    This is used because the metaheuristics are stochastic and the
    result can change from run to run.

summarize_results(results)
    Takes all trial outputs and turns them into the final summary
    numbers, like the mean cost, cost standard deviation, best trial,
    and mean return/risk metrics when they are available.

plot_results(results)
    Makes simple bar plots from the final summaries.
    It also saves the plot and a simple text summary to the
    results folder.
"""
