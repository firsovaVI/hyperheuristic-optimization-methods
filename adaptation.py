import numpy as np  # Добавляем импорт numpy


def adapt_parameters(F, crossover_prob, population, fitness):
    """
    Адаптация параметров F и crossover_prob на основе дисперсии популяции.

    :param F: Коэффициент масштабирования.
    :param crossover_prob: Вероятность кроссовера.
    :param population: Популяция.
    :param fitness: Значения целевой функции.
    :return: Новые значения F и crossover_prob.
    """
    # Пример простой адаптации: уменьшение F и crossover_prob, если дисперсия мала
    var = np.var(population, axis=0).mean()

    if var < 1e-5:
        F *= 0.9
        crossover_prob *= 0.9

    return F, crossover_prob