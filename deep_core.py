import numpy as np


def initialize_population(pop_size, param_bounds):
    """
    Инициализация популяции случайными значениями параметров.

    :param pop_size: Размер популяции.
    :param param_bounds: Границы параметров (список кортежей (min, max)).
    :return: Популяция (массив размером pop_size x num_params).
    """
    num_params = len(param_bounds)
    population = np.zeros((pop_size, num_params))

    for i in range(pop_size):
        for j in range(num_params):
            population[i, j] = np.random.uniform(param_bounds[j][0], param_bounds[j][1])

    return population


def recombine(population, F, crossover_prob):
    """
    Рекомбинация с использованием дифференциальной эволюции.

    :param population: Текущая популяция.
    :param F: Коэффициент масштабирования.
    :param crossover_prob: Вероятность кроссовера.
    :return: Новая популяция после рекомбинации.
    """
    pop_size, num_params = population.shape
    new_population = np.zeros_like(population)

    for i in range(pop_size):
        # Выбор трех случайных индивидов
        a, b, c = np.random.choice(pop_size, 3, replace=False)

        # Создание мутантного вектора
        mutant = population[a] + F * (population[b] - population[c])

        # Кроссовер
        trial = np.where(np.random.rand(num_params) < crossover_prob, mutant, population[i])

        new_population[i] = trial

    return new_population


def evaluate_population(population, objective_function):
    """
    Оценка популяции с использованием целевой функции.

    :param population: Популяция.
    :param objective_function: Целевая функция.
    :return: Массив значений целевой функции для каждого индивида.
    """
    return np.array([objective_function(ind) for ind in population])


def select(population, new_population, fitness, new_fitness):
    """
    Селекция: выбор лучших индивидов из текущей и новой популяции.

    :param population: Текущая популяция.
    :param new_population: Новая популяция.
    :param fitness: Значения целевой функции для текущей популяции.
    :param new_fitness: Значения целевой функции для новой популяции.
    :return: Новая популяция и соответствующие значения целевой функции.
    """
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