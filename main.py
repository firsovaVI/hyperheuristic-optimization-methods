from optimizer import HybridOptimizer
import numpy as np
import matplotlib.pyplot as plt
import json
import pandas as pd
from matplotlib.gridspec import GridSpec


def parabola_objective(x):
    true_params = [2, -3, 5]  # Истинные параметры параболы (a, b, c)
    return sum((x[i] - true_params[i]) ** 2 for i in range(3))


def plot_optimization_history(log_file):
    """График истории оптимизации"""
    with open(log_file) as f:
        data = [json.loads(line) for line in f]

    iterations = [entry['iteration'] for entry in data]
    fitness = [entry['best_fitness'] for entry in data]
    methods = [0 if entry['method'] == 'DEEP' else 1 for entry in data]

    plt.figure(figsize=(12, 8))
    gs = GridSpec(2, 1, height_ratios=[3, 1])

    ax0 = plt.subplot(gs[0])
    ax0.plot(iterations, fitness, 'b-', label='Лучшее значение функции')
    ax0.set_ylabel('Ошибка')
    ax0.set_title('История оптимизации')
    ax0.grid(True)
    ax0.legend()

    ax1 = plt.subplot(gs[1])
    colors = ['blue' if m == 0 else 'red' for m in methods]
    ax1.scatter(iterations, methods, c=colors, alpha=0.6, s=20)
    ax1.set_yticks([0, 1])
    ax1.set_yticklabels(['DEEP', 'Bandit'])
    ax1.set_xlabel('Итерация')
    ax1.set_ylabel('Метод')
    ax1.grid(True)

    plt.tight_layout()
    plt.savefig('optimization_history.png', dpi=300)
    plt.show()


def plot_parabola_comparison(best_params):
    """Сравнение истинной и оптимизированной парабол"""
    x = np.linspace(-5, 5, 100)
    true_y = 2 * x ** 2 - 3 * x + 5
    optimized_y = best_params[0] * x ** 2 + best_params[1] * x + best_params[2]

    plt.figure(figsize=(10, 6))
    plt.plot(x, true_y, 'b-', label='Истинная парабола: $2x^2 - 3x + 5$')
    plt.plot(x, optimized_y, 'r--',
             label=f'Оптимизированная: ${best_params[0]:.2f}x^2 + {best_params[1]:.2f}x + {best_params[2]:.2f}$')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Сравнение парабол')
    plt.legend()
    plt.grid(True)
    plt.savefig('parabola_comparison.png', dpi=300)
    plt.show()


def plot_parameter_evolution(log_file):
    """График изменения параметров"""
    with open(log_file) as f:
        data = [json.loads(line) for line in f]

    iterations = [entry['iteration'] for entry in data]
    a_vals = [entry['best_params']['a'] for entry in data]
    b_vals = [entry['best_params']['b'] for entry in data]
    c_vals = [entry['best_params']['c'] for entry in data]

    plt.figure(figsize=(12, 6))
    plt.plot(iterations, a_vals, 'r-', label='Параметр a')
    plt.plot(iterations, b_vals, 'g-', label='Параметр b')
    plt.plot(iterations, c_vals, 'b-', label='Параметр c')

    plt.axhline(2, color='r', linestyle='--', alpha=0.3, label='Истинное a=2')
    plt.axhline(-3, color='g', linestyle='--', alpha=0.3, label='Истинное b=-3')
    plt.axhline(5, color='b', linestyle='--', alpha=0.3, label='Истинное c=5')

    plt.xlabel('Итерация')
    plt.ylabel('Значение параметра')
    plt.title('Эволюция параметров параболы')
    plt.legend()
    plt.grid(True)
    plt.savefig('parameter_evolution.png', dpi=300)
    plt.show()


if __name__ == "__main__":
    param_bounds = [(-10, 10), (-10, 10), (-10, 10)]
    log_file = "optimization_log.json"

    optimizer = HybridOptimizer(parabola_objective, param_bounds, log_file)

    try:
        print("Запуск оптимизации...")
        best_params, best_fitness = optimizer.optimize(max_iterations=100)

        print("\nРезультаты оптимизации:")
        print(f"Найденные параметры: a={best_params[0]:.4f}, b={best_params[1]:.4f}, c={best_params[2]:.4f}")
        print(f"Лучшее значение функции: {best_fitness:.6f}")

        plot_optimization_history(log_file)
        plot_parabola_comparison(best_params)
        plot_parameter_evolution(log_file)

    except Exception as e:
        print(f"Ошибка в main: {e}")
