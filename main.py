from optimizer import HybridOptimizer
import numpy as np
import matplotlib.pyplot as plt
import json
import pandas as pd


def parabola_objective(x):
    true_params = [2, -3, 5]
    return sum((x[i] - true_params[i]) ** 2 for i in range(3))


def analyze_results(log_file):
    # Чтение данных
    with open(log_file) as f:
        data = [json.loads(line) for line in f]

    # Анализ использования методов
    methods = [entry['method'] for entry in data]
    method_counts = pd.Series(methods).value_counts()

    print("\n=== Методы оптимизации ===")
    print(method_counts)
    print("\nРаспределение методов:")
    print(method_counts / len(data) * 100)

    # Графики
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # График использования методов
    method_counts.plot(kind='bar', ax=ax1)
    ax1.set_title("Использование методов оптимизации")
    ax1.set_ylabel("Количество итераций")

    # График изменения параметров
    iterations = [entry['iteration'] for entry in data]
    a_values = [entry['best_params']['a'] for entry in data]
    b_values = [entry['best_params']['b'] for entry in data]
    c_values = [entry['best_params']['c'] for entry in data]

    ax2.plot(iterations, a_values, label='a')
    ax2.plot(iterations, b_values, label='b')
    ax2.plot(iterations, c_values, label='c')
    ax2.axhline(2, color='r', linestyle='--', alpha=0.3)
    ax2.axhline(-3, color='g', linestyle='--', alpha=0.3)
    ax2.axhline(5, color='b', linestyle='--', alpha=0.3)
    ax2.set_title("Изменение параметров")
    ax2.set_xlabel("Итерация")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('optimization_stats.png')
    plt.show()


if __name__ == "__main__":
    param_bounds = [(-10, 10), (-10, 10), (-10, 10)]
    log_file = "optimization_log.json"

    optimizer = HybridOptimizer(parabola_objective, param_bounds, log_file)

    try:
        best_params, best_fitness = optimizer.optimize(max_iterations=100)
        print(f"\nОптимальные параметры: a={best_params[0]:.4f}, b={best_params[1]:.4f}, c={best_params[2]:.4f}")
        print(f"Лучшее значение функции: {best_fitness:.6f}")

        analyze_results(log_file)

        # Визуализация парабол
        x = np.linspace(-5, 5, 100)
        plt.figure(figsize=(10, 6))
        plt.plot(x, 2 * x ** 2 - 3 * x + 5, label='Истинная парабола')
        plt.plot(x, best_params[0] * x ** 2 + best_params[1] * x + best_params[2], '--', label='Оптимизированная')
        plt.legend()
        plt.grid()
        plt.title("Сравнение парабол")
        plt.savefig('parabola_comparison.png')
        plt.show()

    except Exception as e:
        print(f"Ошибка: {str(e)}")
