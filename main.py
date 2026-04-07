'''
Project runner

@authors
Aidan Ryther arr9180
Tiffany Lee tl1105
Grayson Siegler grs2116
'''

import os

import matplotlib.pyplot as plt
import numpy as np

from algorithms.de import differential_evolution
from algorithms.ga import genetic_algorithm
from algorithms.mofa import multiobjective_firefly
from algorithms.pso import particle_swarm
from algorithms.sa import simulated_annealing
from cost import normalize_weights
from cost import portfolio_return
from cost import portfolio_risk
from cost import portfolio_cost
from evaluate import plot_results
from evaluate import run_one_trial
from evaluate import run_trials
from evaluate import summarize_results
from load_data import load_all_datasets


"""
run_dataset_experiment(dataset_name, dataset, algorithms):
dataset_name is the dataset label. dataset is one loaded dataset.
algorithms holds the algorithm functions and settings. This runs
one dataset with all algorithms.
"""
def run_dataset_experiment(dataset_name, dataset, algorithms):
    mean_returns = dataset["mean_returns"]
    covariance_matrix = dataset["covariance_matrix"]
    dataset_results = {}

    print("Running dataset:", dataset_name)

    for algorithm_name in algorithms:
        curr_setup = algorithms[algorithm_name]
        curr_algorithm = curr_setup["algorithm"]
        curr_args = curr_setup["build_args"](dataset["n_assets"])

        curr_args["mean_returns"] = mean_returns
        curr_args["covariance_matrix"] = covariance_matrix
        curr_args["max_weight"] = curr_setup["max_weight"]
        curr_args["seed_base"] = curr_setup["seed_base"]

        risk_weight = curr_setup["risk_weight"]
        max_weight = curr_setup["max_weight"]

        cost_fn = lambda x, mean_returns=mean_returns, covariance_matrix=covariance_matrix, risk_weight=risk_weight, max_weight=max_weight: portfolio_cost(
            x,
            mean_returns,
            covariance_matrix,
            risk_weight,
            max_weight=max_weight,
        )

        if algorithm_name == "mofa":
            curr_args["g"] = lambda x, covariance_matrix=covariance_matrix, max_weight=max_weight: portfolio_risk(
                normalize_weights(x, max_weight=max_weight),
                covariance_matrix,
            )
            curr_args["h"] = lambda x, mean_returns=mean_returns, max_weight=max_weight: -portfolio_return(
                normalize_weights(x, max_weight=max_weight),
                mean_returns,
            )

        trial_results = run_trials(
            curr_algorithm,
            curr_args,
            cost_fn,
            curr_setup["n_trials"],
        )

        summary = summarize_results(trial_results)
        summary["dataset_name"] = dataset_name
        summary["algorithm_name"] = algorithm_name
        dataset_results[algorithm_name] = summary

        print("  " + algorithm_name + " mean cost:", summary["cost_mean"])

    return dataset_results


"""
run_all_experiments(data_dir): data_dir is the folder holding the
portfolio files. This runs the full project on all datasets.
"""
def run_all_experiments(data_dir):
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    color_map = {
        "sa": "#c0392b",
        "de": "#2563eb",
        "pso": "#1f9d55",
        "ga": "#7c3aed",
        "mofa": "#f59e0b",
    }

    old_files = [
        "summary.txt",
        "dataset_summary.txt",
        "final_summary.txt",
        "mean_cost_by_algorithm.png",
        "mean_cost_by_dataset.png",
        "overall_mean_cost.png",
        "overall_runtime.png",
        "overall_return_volatility.png",
        "portfolio_risk_return_by_dataset.png",
        "mofa_pareto_front_by_dataset.png",
        "cost_vs_epochs.png",
    ]

    for file_name in old_files:
        curr_path = os.path.join(results_dir, file_name)
        if os.path.exists(curr_path):
            os.remove(curr_path)

    datasets = load_all_datasets(data_dir)

    n_trials = 5
    population_size = 20
    n_epoch = 30
    risk_weight = 1.0
    max_weight = 0.20

    algorithms = {
        "sa": {
            "algorithm": simulated_annealing,
            "n_trials": n_trials,
            "risk_weight": risk_weight,
            "max_weight": max_weight,
            "seed_base": 100,
            "build_args": lambda n_assets: {
                "X": lambda n_assets=n_assets: np.random.random((population_size, n_assets)),
                "n_epoch": n_epoch,
                "step_size": 0.05,
                "temp_init": 1.0,
                "temp_decay": 0.98,
            },
        },
        "de": {
            "algorithm": differential_evolution,
            "n_trials": n_trials,
            "risk_weight": risk_weight,
            "max_weight": max_weight,
            "seed_base": 200,
            "build_args": lambda n_assets: {
                "X": lambda n_assets=n_assets: np.random.random((population_size, n_assets)),
                "n_epoch": n_epoch,
                "diff_weight": 0.5,
                "crossover_prob": 0.9,
            },
        },
        "pso": {
            "algorithm": particle_swarm,
            "n_trials": n_trials,
            "risk_weight": risk_weight,
            "max_weight": max_weight,
            "seed_base": 300,
            "build_args": lambda n_assets: {
                "X": lambda n_assets=n_assets: np.random.random((population_size, n_assets)),
                "velocities": lambda n_assets=n_assets: np.random.normal(0.0, 0.02, (population_size, n_assets)),
                "n_epoch": n_epoch,
                "alpha": 1.0,
                "beta": 1.0,
                "theta": 0.9,
            },
        },
        "ga": {
            "algorithm": genetic_algorithm,
            "n_trials": n_trials,
            "risk_weight": risk_weight,
            "max_weight": max_weight,
            "seed_base": 400,
            "build_args": lambda n_assets: {
                "X": lambda n_assets=n_assets: np.random.random((population_size, n_assets)),
                "n_epoch": n_epoch,
                "mutation_rate": 0.10,
                "crossover_rate": 0.80,
            },
        },
        "mofa": {
            "algorithm": multiobjective_firefly,
            "n_trials": n_trials,
            "risk_weight": risk_weight,
            "max_weight": max_weight,
            "seed_base": 500,
            "build_args": lambda n_assets: {
                "X": lambda n_assets=n_assets: np.random.random((population_size, n_assets)),
                "n_epoch": n_epoch,
                "beta0": 1.0,
                "gamma": 1.0,
                "alpha0": 0.25,
            },
        },
    }

    all_results = {}

    for dataset_name in datasets:
        dataset = datasets[dataset_name]
        dataset_results = run_dataset_experiment(dataset_name, dataset, algorithms)
        all_results[dataset_name] = dataset_results

    # Save the dataset-by-algorithm cost graph and summary text.
    plot_results(all_results)

    summary_path = os.path.join(results_dir, "summary.txt")
    dataset_summary_path = os.path.join(results_dir, "dataset_summary.txt")
    if os.path.exists(summary_path):
        os.replace(summary_path, dataset_summary_path)

    algorithm_names = list(algorithms.keys())
    dataset_names = list(all_results.keys())

    overall_results = {}

    for algorithm_name in algorithm_names:
        cost_vals = np.array([all_results[name][algorithm_name]["cost_mean"] for name in dataset_names], dtype=float)
        return_vals = np.array([all_results[name][algorithm_name]["return_mean"] for name in dataset_names], dtype=float)
        volatility_vals = np.array([all_results[name][algorithm_name]["volatility_mean"] for name in dataset_names], dtype=float)
        runtime_vals = np.array([all_results[name][algorithm_name]["runtime_mean"] for name in dataset_names], dtype=float)

        if len(cost_vals) > 1:
            cost_std = float(np.std(cost_vals, ddof=1))
            return_std = float(np.std(return_vals, ddof=1))
            volatility_std = float(np.std(volatility_vals, ddof=1))
            runtime_std = float(np.std(runtime_vals, ddof=1))
        else:
            cost_std = 0.0
            return_std = 0.0
            volatility_std = 0.0
            runtime_std = 0.0

        overall_results[algorithm_name] = {
            "cost_mean": float(np.mean(cost_vals)),
            "cost_std": cost_std,
            "return_mean": float(np.mean(return_vals)),
            "return_std": return_std,
            "volatility_mean": float(np.mean(volatility_vals)),
            "volatility_std": volatility_std,
            "runtime_mean": float(np.mean(runtime_vals)),
            "runtime_std": runtime_std,
        }

    # Graph 2: overall mean cost by algorithm.
    cost_order = list(algorithm_names)
    cost_order.sort(key=lambda name: overall_results[name]["cost_mean"])

    cost_means = [overall_results[name]["cost_mean"] for name in cost_order]
    cost_stds = [overall_results[name]["cost_std"] for name in cost_order]
    cost_colors = [color_map.get(name, "#bfc5cc") for name in cost_order]

    fig_cost, ax_cost = plt.subplots(figsize=(8, 4))
    ax_cost.bar(cost_order, cost_means, yerr=cost_stds, capsize=4, color=cost_colors, edgecolor="black")
    ax_cost.set_title("Overall Mean Cost by Algorithm")
    ax_cost.set_ylabel("Mean Cost")
    ax_cost.set_xlabel("Algorithm")
    ax_cost.grid(axis="y", linestyle="--", alpha=0.3)
    ax_cost.set_axisbelow(True)
    fig_cost.tight_layout()
    fig_cost.savefig(os.path.join(results_dir, "overall_mean_cost.png"), dpi=200)

    # Graph 3: overall runtime by algorithm.
    runtime_order = list(algorithm_names)
    runtime_order.sort(key=lambda name: overall_results[name]["runtime_mean"])

    runtime_means = [overall_results[name]["runtime_mean"] for name in runtime_order]
    runtime_stds = [overall_results[name]["runtime_std"] for name in runtime_order]
    runtime_colors = [color_map.get(name, "#bfc5cc") for name in runtime_order]

    fig_runtime, ax_runtime = plt.subplots(figsize=(8, 4))
    ax_runtime.bar(runtime_order, runtime_means, yerr=runtime_stds, capsize=4, color=runtime_colors, edgecolor="black")
    ax_runtime.set_title("Overall Mean Runtime by Algorithm")
    ax_runtime.set_ylabel("Seconds")
    ax_runtime.set_xlabel("Algorithm")
    ax_runtime.grid(axis="y", linestyle="--", alpha=0.3)
    ax_runtime.set_axisbelow(True)
    fig_runtime.tight_layout()
    fig_runtime.savefig(os.path.join(results_dir, "overall_runtime.png"), dpi=200)

    # Graph 4: overall mean return versus volatility.
    fig_metrics, ax_metrics = plt.subplots(figsize=(8, 5))

    for algorithm_name in algorithm_names:
        curr_summary = overall_results[algorithm_name]
        curr_x = curr_summary["volatility_mean"]
        curr_y = curr_summary["return_mean"]
        curr_color = color_map.get(algorithm_name, "#bfc5cc")

        ax_metrics.scatter(
            curr_x,
            curr_y,
            s=140,
            color=curr_color,
            edgecolor="black",
            linewidth=1.0,
            label=algorithm_name.upper(),
        )
        ax_metrics.annotate(
            algorithm_name.upper(),
            (curr_x, curr_y),
            textcoords="offset points",
            xytext=(6, 6),
        )

    ax_metrics.set_title("Overall Risk-Return by Algorithm")
    ax_metrics.set_xlabel("Mean Volatility")
    ax_metrics.set_ylabel("Mean Return")
    ax_metrics.grid(True, linestyle="--", alpha=0.3)
    fig_metrics.tight_layout()
    fig_metrics.savefig(os.path.join(results_dir, "overall_return_volatility.png"), dpi=200)

    # Graph 5: assets and best portfolios in risk-return space.
    n_datasets = len(dataset_names)
    n_cols = 2
    n_rows = int(np.ceil(n_datasets / n_cols))
    fig_map, axes_map = plt.subplots(n_rows, n_cols, figsize=(10, 4 * n_rows))
    axes_map = np.asarray(axes_map).reshape(-1)

    for i in range(len(dataset_names)):
        dataset_name = dataset_names[i]
        curr_ax = axes_map[i]
        curr_data = datasets[dataset_name]

        asset_volatility = curr_data["std_devs"]
        asset_return = curr_data["mean_returns"]

        curr_ax.scatter(
            asset_volatility,
            asset_return,
            s=28,
            color="#cfd4da",
            edgecolor="none",
            alpha=0.9,
            label="Assets",
        )

        for algorithm_name in algorithm_names:
            curr_summary = all_results[dataset_name][algorithm_name]
            curr_ax.scatter(
                curr_summary["volatility_best"],
                curr_summary["return_best"],
                s=90,
                color=color_map.get(algorithm_name, "#bfc5cc"),
                edgecolor="black",
                linewidth=0.8,
            )

        curr_ax.set_title(dataset_name + " Risk-Return")
        curr_ax.set_xlabel("Volatility")
        curr_ax.set_ylabel("Return")
        curr_ax.grid(True, linestyle="--", alpha=0.3)

    for i in range(len(dataset_names), len(axes_map)):
        axes_map[i].axis("off")

    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#cfd4da", markersize=7, label="Assets"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color_map["sa"], markeredgecolor="black", markersize=8, label="SA"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color_map["de"], markeredgecolor="black", markersize=8, label="DE"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color_map["pso"], markeredgecolor="black", markersize=8, label="PSO"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color_map["ga"], markeredgecolor="black", markersize=8, label="GA"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color_map["mofa"], markeredgecolor="black", markersize=8, label="MOFA"),
    ]
    fig_map.legend(handles=legend_handles, loc="upper center", ncol=6, frameon=False)
    fig_map.tight_layout(rect=[0, 0, 1, 0.95])
    fig_map.savefig(os.path.join(results_dir, "portfolio_risk_return_by_dataset.png"), dpi=200)

    # Graph 6: MOFA Pareto front for each dataset.
    fig_mofa, axes_mofa = plt.subplots(n_rows, n_cols, figsize=(10, 4 * n_rows))
    axes_mofa = np.asarray(axes_mofa).reshape(-1)

    for i in range(len(dataset_names)):
        dataset_name = dataset_names[i]
        curr_ax = axes_mofa[i]
        curr_data = datasets[dataset_name]
        mofa_summary = all_results[dataset_name]["mofa"]
        best_trial_idx = mofa_summary["best_trial"] - 1
        mofa_trial = mofa_summary["trial_results"][best_trial_idx]
        front_X = mofa_trial["final_population"]

        asset_volatility = curr_data["std_devs"]
        asset_return = curr_data["mean_returns"]
        front_weights = normalize_weights(front_X, max_weight=max_weight)
        front_return = portfolio_return(front_weights, curr_data["mean_returns"])
        front_volatility = np.sqrt(portfolio_risk(front_weights, curr_data["covariance_matrix"]))

        curr_ax.scatter(
            asset_volatility,
            asset_return,
            s=24,
            color="#d7dbe0",
            edgecolor="none",
            alpha=0.9,
        )
        curr_ax.scatter(
            front_volatility,
            front_return,
            s=34,
            color=color_map["mofa"],
            edgecolor="none",
            alpha=0.8,
        )
        curr_ax.scatter(
            mofa_summary["volatility_best"],
            mofa_summary["return_best"],
            s=120,
            color=color_map["mofa"],
            edgecolor="black",
            linewidth=1.0,
        )
        curr_ax.set_title(dataset_name + " MOFA Pareto Front")
        curr_ax.set_xlabel("Volatility")
        curr_ax.set_ylabel("Return")
        curr_ax.grid(True, linestyle="--", alpha=0.3)

    for i in range(len(dataset_names), len(axes_mofa)):
        axes_mofa[i].axis("off")

    mofa_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#d7dbe0", markersize=7, label="Assets"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color_map["mofa"], markersize=7, label="MOFA Front"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color_map["mofa"], markeredgecolor="black", markersize=9, label="MOFA Best"),
    ]
    fig_mofa.legend(handles=mofa_handles, loc="upper center", ncol=3, frameon=False)
    fig_mofa.tight_layout(rect=[0, 0, 1, 0.95])
    fig_mofa.savefig(os.path.join(results_dir, "mofa_pareto_front_by_dataset.png"), dpi=200)

    # Graph 7: mean cost as the epoch budget grows.
    epoch_points = [1, 5, 10, 15, 20, 25, n_epoch]
    epoch_points = sorted(set(epoch_points))
    cost_lines = {}

    for algorithm_name in algorithm_names:
        cost_lines[algorithm_name] = []
        curr_setup = algorithms[algorithm_name]

        for epoch_count in epoch_points:
            epoch_costs = []

            for i, dataset_name in enumerate(dataset_names):
                curr_data = datasets[dataset_name]
                mean_returns = curr_data["mean_returns"]
                covariance_matrix = curr_data["covariance_matrix"]
                max_weight = curr_setup["max_weight"]
                risk_weight = curr_setup["risk_weight"]

                curr_args = curr_setup["build_args"](curr_data["n_assets"])
                curr_args["n_epoch"] = epoch_count
                curr_args["mean_returns"] = mean_returns
                curr_args["covariance_matrix"] = covariance_matrix
                curr_args["max_weight"] = max_weight
                curr_args["seed"] = curr_setup["seed_base"] + i

                cost_fn = lambda x, mean_returns=mean_returns, covariance_matrix=covariance_matrix, risk_weight=risk_weight, max_weight=max_weight: portfolio_cost(
                    x,
                    mean_returns,
                    covariance_matrix,
                    risk_weight,
                    max_weight=max_weight,
                )

                if algorithm_name == "mofa":
                    curr_args["g"] = lambda x, covariance_matrix=covariance_matrix, max_weight=max_weight: portfolio_risk(
                        normalize_weights(x, max_weight=max_weight),
                        covariance_matrix,
                    )
                    curr_args["h"] = lambda x, mean_returns=mean_returns, max_weight=max_weight: -portfolio_return(
                        normalize_weights(x, max_weight=max_weight),
                        mean_returns,
                    )

                trial_result = run_one_trial(curr_setup["algorithm"], curr_args, cost_fn)
                epoch_costs.append(trial_result["best_cost"])

            cost_lines[algorithm_name].append(float(np.mean(epoch_costs)))

    fig_lines, ax_lines = plt.subplots(figsize=(8, 5))

    for algorithm_name in algorithm_names:
        ax_lines.plot(
            epoch_points,
            cost_lines[algorithm_name],
            marker="o",
            linewidth=2,
            color=color_map.get(algorithm_name, "#bfc5cc"),
            label=algorithm_name.upper(),
        )

    ax_lines.set_title("Mean Cost vs Epochs")
    ax_lines.set_xlabel("Epochs")
    ax_lines.set_ylabel("Mean Cost")
    ax_lines.grid(True, linestyle="--", alpha=0.3)
    ax_lines.legend(frameon=False)
    fig_lines.tight_layout()
    fig_lines.savefig(os.path.join(results_dir, "cost_vs_epochs.png"), dpi=200)

    summary_lines = []
    summary_lines.append("Project Settings")
    summary_lines.append("")
    summary_lines.append("n_trials: " + str(n_trials))
    summary_lines.append("population_size: " + str(population_size))
    summary_lines.append("n_epoch: " + str(n_epoch))
    summary_lines.append("risk_weight: " + str(risk_weight))
    summary_lines.append("max_weight: " + str(max_weight))
    summary_lines.append("")
    summary_lines.append("Overall Means by Algorithm")
    summary_lines.append("")

    for algorithm_name in cost_order:
        curr_summary = overall_results[algorithm_name]
        summary_lines.append(algorithm_name)
        summary_lines.append("  mean cost: " + str(curr_summary["cost_mean"]))
        summary_lines.append("  mean runtime: " + str(curr_summary["runtime_mean"]))
        summary_lines.append("  mean return: " + str(curr_summary["return_mean"]))
        summary_lines.append("  mean volatility: " + str(curr_summary["volatility_mean"]))
        summary_lines.append("")

    summary_lines.append("Per Dataset Results")
    summary_lines.append("")

    for dataset_name in dataset_names:
        summary_lines.append(dataset_name)
        summary_lines.append("")

        dataset_results = all_results[dataset_name]

        for algorithm_name in algorithm_names:
            curr_summary = dataset_results[algorithm_name]
            summary_lines.append("  " + algorithm_name)
            summary_lines.append("    cost mean: " + str(curr_summary["cost_mean"]))
            summary_lines.append("    cost std: " + str(curr_summary["cost_std"]))
            summary_lines.append("    runtime mean: " + str(curr_summary["runtime_mean"]))
            summary_lines.append("    return mean: " + str(curr_summary["return_mean"]))
            summary_lines.append("    risk mean: " + str(curr_summary["risk_mean"]))
            summary_lines.append("    volatility mean: " + str(curr_summary["volatility_mean"]))
            summary_lines.append("    sharpe_like mean: " + str(curr_summary["sharpe_like_mean"]))
            summary_lines.append("")

        summary_lines.append("")

    final_summary_path = os.path.join(results_dir, "final_summary.txt")
    with open(final_summary_path, "w") as outFile:
        outFile.write("\n".join(summary_lines))

    print("")
    print("Saved results to:", results_dir)
    print("Saved graphs:")
    print("  results/mean_cost_by_dataset.png")
    print("  results/overall_mean_cost.png")
    print("  results/overall_runtime.png")
    print("  results/overall_return_volatility.png")
    print("  results/portfolio_risk_return_by_dataset.png")
    print("  results/mofa_pareto_front_by_dataset.png")
    print("  results/cost_vs_epochs.png")

    return all_results


"""
main(): This starts the full project run.
"""
def main():
    project_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_dir)

    data_dir = os.path.join(project_dir, "data")
    return run_all_experiments(data_dir)


"""
Main summary:

run_dataset_experiment(dataset_name, dataset, algorithms)
    Runs one dataset with all of the algorithms.
    It builds the cost function for that dataset, runs the trials,
    and stores the final summary for each algorithm.

run_all_experiments(data_dir)
    Runs the full project on all portfolio datasets.
    It loads the data, sets the algorithm hyper-parameters, saves
    the text summaries, and saves the final graphs.

main()
    Starts the whole project from the project folder so the data path
    and results folder always line up correctly.
"""


if __name__ == "__main__":
    main()
