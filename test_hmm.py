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


def test_update_emissions():
    hmm = HiddenMarkovModel(n_states=2)

    # Инициализируем историю для стабильности Viterbi
    hmm.f_best_history = [1.0, 0.5]
    hmm.variance_history = [0.8, 0.4]

    # Простые наблюдения, которые точно попадут в разные состояния
    observations = [0, 1, 0, 3, 2]  # Используем маленькие числа для ясности

    initial_counts = hmm.emission_counts.copy()
    hmm.update_emissions(observations)

    # Проверяем общее увеличение счетчиков
    assert np.all(hmm.emission_counts >= initial_counts)

    # Проверяем конкретные увеличения (адаптивно)
    for obs in observations:
        state = hmm.viterbi_algorithm([obs])[0]  # Получаем состояние для этого наблюдения
        assert hmm.emission_counts[state, obs] >= initial_counts[state, obs] + 1


def test_update_transitions():
    hmm = HiddenMarkovModel(n_states=2)

    # Искусственно изменяем начальные вероятности для теста
    hmm.transition_probs = np.array([[0.9, 0.1], [0.1, 0.9]])
    initial_transitions = hmm.transition_probs.copy()

    hmm.update_transitions()

    # Проверяем что вероятности изменились (с учетом искусственных данных)
    assert not np.allclose(hmm.transition_probs, initial_transitions, atol=0.01)
    assert np.allclose(hmm.transition_probs.sum(axis=1), 1.0)

def test_full_update_cycle():
    hmm = HiddenMarkovModel(n_states=2)
    f_history = np.linspace(1.0, 0.1, 16)
    var_history = np.linspace(0.1, 1.0, 16)

    hmm.update_with_optimizer_data(f_history, var_history)

    assert len(hmm.observation_history) == 16
    assert not np.allclose(hmm.emission_probs, 1 / hmm.n_observations)
    assert hmm.emission_counts.sum() > hmm.n_states * hmm.n_observations  # Изначальные + новые


if __name__ == "__main__":
    pytest.main(["-v", "test_hmm.py"])
