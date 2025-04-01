import numpy as np
from deep_core import *
from bandit import BernoulliBandit
from solver import EpsilonGreedy
from hmm import HiddenMarkovModel
import json
import os


class HybridOptimizer:
    def __init__(self, objective_function, param_bounds, log_file="optimization_log.json"):
        self.param_bounds = param_bounds  # Явно сохраняем param_bounds
        self.objective_function = lambda x: objective_function(transform_u_to_q(x, param_bounds))
        self.hmm = HiddenMarkovModel(n_states=2, n_observations=2)
        self.bandit = EpsilonGreedy(BernoulliBandit(4))
        self.current_method = 0
        self.log_file = log_file
        self.history = []

        if os.path.exists(self.log_file):
            os.remove(self.log_file)

    def _save_iteration_data(self, iteration, population, fitness):
        try:
            best_idx = np.argmin(fitness)
            best_params = transform_u_to_q(population[best_idx], self.param_bounds)

            data = {
                "iteration": iteration,
                "best_params": {
                    "a": float(best_params[0]),
                    "b": float(best_params[1]),
                    "c": float(best_params[2])
                },
                "best_fitness": float(fitness[best_idx]),
                "method": "DEEP" if self.current_method == 0 else "Bandit"
            }

            with open(self.log_file, 'a') as f:
                f.write(json.dumps(data) + '\n')
        except Exception as e:
            print(f"Error saving iteration data: {e}")

    def optimize(self, max_iterations=100):
        try:
            population = initialize_population(50, self.param_bounds)
            fitness = evaluate_population(population, self.objective_function)

            observations = []

            for iteration in range(max_iterations):
                if iteration > 10:
                    state_sequence = self.hmm.viterbi_algorithm(observations)
                    self.current_method = state_sequence[-1]
                else:
                    self.current_method = np.random.randint(0, 2)

                if self.current_method == 0:
                    new_population = recombine(population, 0.5, 0.7)
                else:
                    strategy_idx = self.bandit.run_one_step()
                    new_population = population.copy()
                    mask = np.random.rand(*new_population.shape) < 0.2
                    new_population += mask * np.random.normal(0, 0.2, size=new_population.shape)

                new_fitness = evaluate_population(new_population, self.objective_function)
                population, fitness = select(population, new_population, fitness, new_fitness)

                reward = 1 if np.min(new_fitness) < np.min(fitness) - 1e-6 else 0
                observations.append(reward)

                if iteration % 20 == 0 and iteration > 0:
                    self.hmm.update_model(self.current_method, reward)

                self._save_iteration_data(iteration, population, fitness)

                if iteration % 10 == 0:
                    best_idx = np.argmin(fitness)
                    print(f"Iter {iteration}: Method={'DEEP' if self.current_method == 0 else 'Bandit'}, "
                          f"Params={transform_u_to_q(population[best_idx], self.param_bounds)}, "
                          f"Fitness={fitness[best_idx]:.4f}")

            best_idx = np.argmin(fitness)
            if len(fitness) == 0:
                return np.zeros(len(self.param_bounds)), float('inf')
            return transform_u_to_q(population[best_idx], self.param_bounds), fitness[best_idx]

        except Exception as e:
            print(f"Error in optimize(): {e}")
            return np.zeros(len(self.param_bounds)), float('inf')
