import numpy as np
from optimizer import HybridOptimizer
import matplotlib.pyplot as plt


# Целевая функция - парабола с шумом
def parabola(x, a=1, b=0, c=0, noise_std=0.1):
    """Парабола вида y = a*x^2 + b*x + c + шум"""
    noise = np.random.normal(scale=noise_std)
    return a * x[0] ** 2 + b * x[0] + c + noise


# Функция для оптимизации
def objective_function(x):
    """Целевая функция для оптимизации (минимизация)"""
    # Истинные параметры параболы (которые мы хотим найти)
    true_a, true_b, true_c = 2, -3, 5
    # Вычисляем MSE между текущими параметрами и целевыми
    return (x[0] - true_a) ** 2 + (x[1] - true_b) ** 2 + (x[2] - true_c) ** 2


if __name__ == "__main__":
    # Границы параметров [a, b, c]
    param_bounds = [(-10, 10), (-10, 10), (-10, 10)]

    # Создаем оптимизатор
    optimizer = HybridOptimizer(objective_function, param_bounds)

    # Запускаем оптимизацию
    best_params, best_fitness = optimizer.optimize(max_iterations=100)

    print("\nРезультаты оптимизации:")
    print(f"Найденные параметры: a={best_params[0]:.3f}, b={best_params[1]:.3f}, c={best_params[2]:.3f}")
    print(f"Значение целевой функции: {best_fitness:.6f}")

    # Визуализация
    x = np.linspace(-5, 5, 100)
    true_y = 2 * x ** 2 - 3 * x + 5  # Истинная парабола
    optimized_y = best_params[0] * x ** 2 + best_params[1] * x + best_params[2]  # Найденная парабола

    plt.figure(figsize=(10, 6))
    plt.plot(x, true_y, label="Истинная парабола: $2x^2 - 3x + 5$")
    plt.plot(x, optimized_y, '--',
             label=f"Оптимизированная: ${best_params[0]:.2f}x^2 + {best_params[1]:.2f}x + {best_params[2]:.2f}$")
    plt.scatter([0], [parabola([0])], color='red', label="Точки данных с шумом")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Оптимизация параметров параболы")
    plt.legend()
    plt.grid(True)
    plt.show()
