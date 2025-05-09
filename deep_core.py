import numpy as np

def transform_u_to_q(u, param_bounds):
    try:
        q = np.zeros_like(u)
        for i in range(len(u)):
            alpha = (param_bounds[i][1] + param_bounds[i][0]) / 2
            beta = (param_bounds[i][1] - param_bounds[i][0]) / 2
            q[i] = alpha + beta * np.tanh(u[i])
        return q
    except Exception as e:
        print(f"Error in transform_u_to_q: {e}")
        return np.zeros_like(u)

def initialize_population(pop_size, param_bounds):
    try:
        num_params = len(param_bounds)
        return np.random.uniform(-1, 1, size=(pop_size, num_params))
    except Exception as e:
        print(f"Error in initialize_population: {e}")
        return np.random.uniform(-1, 1, size=(pop_size, 3))  # fallback

def recombine(population, F, crossover_prob):
    try:
        pop_size, num_params = population.shape
        new_population = np.zeros_like(population)
        for i in range(pop_size):
            a, b, c = np.random.choice(pop_size, 3, replace=False)
            mutant = population[a] + F * (population[b] - population[c])
            trial = np.where(np.random.rand(num_params) < crossover_prob, mutant, population[i])
            new_population[i] = trial
        return new_population
    except Exception as e:
        print(f"Error in recombine: {e}")
        return population.copy()

def evaluate_population(population, objective_function):
    try:
        return np.array([objective_function(ind) for ind in population])
    except Exception as e:
        print(f"Error in evaluate_population: {e}")
        return np.ones(len(population)) * float('inf')

def select(population, new_population, fitness, new_fitness):
    try:
        pop_size = population.shape[0]
        selected_population = np.zeros_like(population)
        selected_fitness = np.zeros_like(fitness)
        for i in range(pop_size):
            if new_fitness[i] < fitness[i]:
                selected_population[i] = new_population[i]
                selected_fitness[i] = new_fitness[i]
            else:
                selected_population[i] = population[i]
                selected_fitness[i] = fitness[i]
        return selected_population, selected_fitness
    except Exception as e:
        print(f"Error in select: {e}")
        return population, fitness
