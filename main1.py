import matplotlib  # noqa
matplotlib.use('Agg')  # noqa

import matplotlib.pyplot as plt
import numpy as np

from bandit import BernoulliBandit
from solver import Solver, EpsilonGreedy, UCB1, BayesianUCB, ThompsonSampling


def plot_results(solvers, solver_names, figname):
    """
    Plot the results by multi-armed bandit solvers.

    Args:
        solvers (list<Solver>): All of them should have been fitted.
        solver_names (list<str>): Names of the solvers.
        figname (str): Name of the output figure file.
    """
    assert len(solvers) == len(solver_names)
    assert all(isinstance(s, Solver) for s in solvers)
    assert all(len(s.regrets) > 0 for s in solvers)

    b = solvers[0].bandit

    # Create a figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    fig.subplots_adjust(bottom=0.3, wspace=0.3)

    # Subplot 1: Cumulative Regret
    for i, s in enumerate(solvers):
        ax1.plot(range(len(s.regrets)), s.regrets, label=solver_names[i], linewidth=2)
    ax1.set_xlabel('Time Step', fontsize=12)
    ax1.set_ylabel('Cumulative Regret', fontsize=12)
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Subplot 2: Estimated Probabilities
    sorted_indices = sorted(range(b.n), key=lambda x: b.probas[x])
    ax2.plot(range(b.n), [b.probas[x] for x in sorted_indices], 'k--', markersize=12, label='True Probabilities')
    for s in solvers:
        ax2.plot(range(b.n), [s.estimated_probas[x] for x in sorted_indices], 'x', markeredgewidth=2, label=f'{solver_names[solvers.index(s)]} Estimates')
    ax2.set_xlabel('Actions Sorted by True Probability', fontsize=12)
    ax2.set_ylabel('Estimated Probability', fontsize=12)
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.6)

    # Subplot 3: Action Counts
    for s in solvers:
        ax3.plot(range(b.n), np.array(s.counts) / float(len(solvers[0].regrets)), drawstyle='steps', lw=2, label=solver_names[solvers.index(s)])
    ax3.set_xlabel('Actions', fontsize=12)
    ax3.set_ylabel('Fraction of Trials', fontsize=12)
    ax3.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=10)
    ax3.grid(True, linestyle='--', alpha=0.6)

    # Save the figure
    plt.savefig(figname, dpi=300, bbox_inches='tight')
    plt.close()


def experiment(K, N, seed=None):
    """
    Run a small experiment on solving a Bernoulli bandit with K slot machines,
    each with a randomly initialized reward probability.

    Args:
        K (int): Number of slot machines.
        N (int): Number of time steps to try.
        seed (int, optional): Random seed for reproducibility.
    """
    # Set random seed for reproducibility
    if seed is not None:
        np.random.seed(seed)

    # Initialize the bandit with random probabilities
    b = BernoulliBandit(K)
    print("Randomly generated Bernoulli bandit has reward probabilities:\n", b.probas)
    print(f"The best machine has index: {np.argmax(b.probas)} and probability: {np.max(b.probas):.4f}")

    # Define solvers to test
    test_solvers = [
        EpsilonGreedy(b, 0.01),  # ε-Greedy with ε=0.01
        UCB1(b),  # UCB1 algorithm
        BayesianUCB(b, 3, 1, 1),  # Bayesian UCB with c=3, init_a=1, init_b=1
        ThompsonSampling(b, 1, 1)  # Thompson Sampling with init_a=1, init_b=1
    ]
    solver_names = [
        r'$\epsilon$-Greedy',
        'UCB1',
        'Bayesian UCB',
        'Thompson Sampling'
    ]

    # Run each solver for N steps
    for s in test_solvers:
        s.run(N)

    # Plot and save results
    plot_results(test_solvers, solver_names, f"results_K{K}_N{N}.png")


if __name__ == '__main__':
    # Run the experiment with K=10 arms and N=5000 steps
    experiment(10, 5000, seed=42)  # Set seed for reproducibility