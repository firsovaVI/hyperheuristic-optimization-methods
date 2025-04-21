import pytest
import numpy as np
from hmm import HiddenMarkovModel


def test_hmm_initialization():
    hmm = HiddenMarkovModel(n_states=2)
    assert hmm.n_states == 2
    assert hmm.observation_levels == 8
    assert hmm.n_observations == 64
    assert hmm.transition_probs.shape == (2, 2)
    assert hmm.emission_probs.shape == (2, 64)
    assert np.allclose(hmm.transition_probs.sum(axis=1), 1.0)
    assert np.allclose(hmm.emission_probs.sum(axis=1), 1.0)


def test_discretize_observation():
    hmm = HiddenMarkovModel(n_states=2)
    hmm.f_best_history = [1.0, 0.5, 0.2]
    hmm.variance_history = [0.8, 0.4, 0.1]

    # Проверяем диапазон
    obs = hmm.discretize_observation(0.5, 0.4)
    assert 0 <= obs < hmm.n_observations

    # Проверяем минимальное значение
    assert hmm.discretize_observation(0, 0) == 0

    # Проверяем максимальное значение при максимальных входах
    max_obs = hmm.discretize_observation(1.0, 0.8)
    assert max_obs == hmm.n_observations - 1


def test_update_transitions():
    hmm = HiddenMarkovModel(n_states=2)

    # Явно задаем тестовые данные
    hmm.fixed_obs_sequence = np.array([0, 0, 0, 1, 1, 1])  # Четкая последовательность
    hmm.emission_probs = np.array([
        [0.9, 0.1],  # Состояние 0 чаще выдает наблюдение 0
        [0.1, 0.9]  # Состояние 1 чаще выдает наблюдение 1
    ])

    # Сохраняем начальные вероятности (должны быть случайными после инициализации)
    initial_transitions = hmm.transition_probs.copy()

    # Выполняем обновление
    hmm.update_transitions()

    # Проверяем что вероятности изменились
    assert not np.allclose(hmm.transition_probs, initial_transitions, atol=0.1), \
        f"Transitions didn't change. Before: {initial_transitions}, after: {hmm.transition_probs}"

    # Проверяем нормализацию
    assert np.allclose(hmm.transition_probs.sum(axis=1), 1.0, atol=1e-6)

def test_full_update_cycle():
    hmm = HiddenMarkovModel(n_states=2)

    # Инициализируем историю для стабильности
    initial_counts = hmm.emission_counts.sum()

    # Генерируем тестовые данные
    f_history = np.linspace(1.0, 0.1, 16)
    var_history = np.linspace(0.1, 1.0, 16)

    # Выполняем обновление
    hmm.update_with_optimizer_data(f_history, var_history)

    # Проверяем что наблюдения добавлены
    assert len(hmm.observation_history) == 16

    # Проверяем что счетчики увеличились (но не обязательно сильно)
    assert hmm.emission_counts.sum() > initial_counts

    # Проверяем что вероятности нормализованы
    assert np.allclose(hmm.emission_probs.sum(axis=1), 1.0, atol=1e-6)


def test_emission_update_with_counts():
    hmm = HiddenMarkovModel(n_states=2)
    initial_counts = hmm.emission_counts.copy()

    # Генерируем тестовые данные
    f_history = np.linspace(1.0, 0.1, 10)
    var_history = np.linspace(0.1, 1.0, 10)

    hmm.update_with_optimizer_data(f_history, var_history)

    # Проверяем, что счетчики увеличились
    assert np.any(hmm.emission_counts > initial_counts)

    # Проверяем, что вероятности нормализованы
    assert np.allclose(hmm.emission_probs.sum(axis=1), 1.0)

    # Проверяем, что n_observations равно квадрату levels
    assert hmm.n_observations == hmm.observation_levels ** 2


if __name__ == "__main__":
    pytest.main(["-v", "test_hmm.py"])
