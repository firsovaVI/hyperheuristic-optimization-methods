import numpy as np

from deep_algorithm import deep_algorithm

# Пример целевой функции (сферическая функция)
def objective_function(x):
    return np.sum(x**2)

# Границы параметров (для каждого параметра заданы min и max)
param_bounds = [(-5, 5), (-5, 5), (-5, 5)]

# Запуск алгоритма DEEP
best_solution, best_fitness = deep_algorithm(objective_function, param_bounds, pop_size=50, max_generations=100)

print("Best Solution:", best_solution)
print("Best Fitness:", best_fitness)