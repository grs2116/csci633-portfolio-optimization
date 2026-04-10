# CSCI 633 Portfolio Optimization

This project studies portfolio optimization with nature-inspired metaheuristics.  
The code compares five algorithms on five OR-Library portfolio benchmark datasets:

- Simulated Annealing (`SA`)
- Differential Evolution (`DE`)
- Particle Swarm Optimization (`PSO`)
- Genetic Algorithm (`GA`)
- Multiobjective Firefly Algorithm (`MOFA`)

The project uses a Markowitz-style mean-variance objective with portfolio constraints:

- long-only weights
- fully invested portfolio
- optional max weight per asset

## Authors

- Aidan Ryther
- Tiffany Lee
- Grayson Siegler

## What The Code Does

The project loads benchmark portfolio datasets, runs each algorithm multiple times, scores the portfolios with the same cost function, summarizes the results, and saves graphs and text summaries in the `results/` folder.

The shared portfolio cost is:

`cost(w) = risk_weight * risk(w) - return(w)`

where:

- `return(w) = w^T * mu`
- `risk(w) = w^T * Sigma * w`

Lower cost is better.

## Requirements

The code is written in Python and uses:

- Python 3
- `numpy`
- `matplotlib`

This project was run with:

- Python `3.12.11`

If needed, install the main packages with:

```bash
pip install numpy matplotlib
```

## Project Files

### Main Files

- `main.py`
  Runs the full experiment on all datasets and saves summaries and graphs.

- `load_data.py`
  Loads the OR-Library portfolio benchmark files and builds covariance matrices.

- `cost.py`
  Handles portfolio normalization, return, risk, and the shared cost function.

- `evaluate.py`
  Runs trials, times the runs, summarizes results, and supports plotting.

### Algorithm Files

- `algorithms/sa.py`
  Simulated annealing

- `algorithms/de.py`
  Differential evolution

- `algorithms/pso.py`
  Particle swarm optimization

- `algorithms/ga.py`
  Genetic algorithm

- `algorithms/mofa.py`
  Multiobjective firefly algorithm using Pareto dominance

### Data Files

- `data/port1.txt` through `data/port5.txt`
  The five OR-Library portfolio benchmark problems used in the experiments.

## How To Run The Project

From the project folder:

```bash
cd /Users/aidanryther/Desktop/Bio/csci633-portfolio-optimization
python main.py
```

If `matplotlib` gives a cache warning on your machine, use:

```bash
cd /Users/aidanryther/Desktop/Bio/csci633-portfolio-optimization
MPLCONFIGDIR=/tmp/mpl python main.py
```

## What Gets Saved

Running `main.py` creates or updates the `results/` folder.

Current outputs:

- `results/dataset_summary.txt`
- `results/final_summary.txt`
- `results/mean_cost_by_dataset.png`
- `results/overall_mean_cost.png`
- `results/overall_runtime.png`
- `results/overall_return_volatility.png`
- `results/portfolio_risk_return_by_dataset.png`
- `results/mofa_pareto_front_by_dataset.png`
- `results/cost_vs_epochs.png`

## How To Reproduce The Current Results

The code already uses fixed seeds inside `main.py`, so running the project again with the same settings will reproduce the same experiment setup.

The default settings are currently in `main.py`:

- `n_trials = 5`
- `population_size = 20`
- `n_epoch = 30`
- `risk_weight = 1.0`
- `max_weight = 0.20`

To reproduce the current results:

1. Use the existing `data/port1.txt` through `data/port5.txt` files.
2. Leave the settings in `main.py` unchanged.
3. Run `python main.py` from the project folder.
4. Read the saved summaries in `results/final_summary.txt` and `results/dataset_summary.txt`.

## How To Change The Experiment

If you want to explore more cases, the simplest place to edit is `main.py`.

You can change:

- number of trials
- population size
- epoch count
- cost tradeoff through `risk_weight`
- max portfolio weight through `max_weight`
- algorithm hyperparameters inside the `algorithms` dictionary
