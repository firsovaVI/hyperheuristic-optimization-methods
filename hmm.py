import numpy as np
from scipy.stats import norm


class HiddenMarkovModel:
    def __init__(self, n_states, observation_levels=8, model_length=16):
        """
        Инициализация скрытой марковской модели

        Параметры:
            n_states: количество скрытых состояний
            observation_levels: количество уровней дискретизации (по умолчанию 8)
            model_length: длина последовательности для обучения
        """
        self.n_states = n_states
        self.observation_levels = observation_levels
        self.n_observations = observation_levels ** 2
        self.model_length = model_length

        # Инициализация параметров модели
        self.transition_probs = np.ones((n_states, n_states)) / n_states
        self.initial_probs = np.ones(n_states) / n_states
        self.emission_probs = np.ones((n_states, self.n_observations)) / self.n_observations

        # Для хранения статистики
        self.emission_counts = np.ones((n_states, self.n_observations))
        self.transition_counts = np.ones((n_states, n_states))

        # Фиксированная последовательность для обучения
        self._initialize_smooth_sequence()

        # История наблюдений
        self.observation_history = []
        self.state_history = []
        self.f_best_history = []
        self.variance_history = []

    def _initialize_smooth_sequence(self):
        """Инициализация плавной последовательности для обучения"""
        x = np.linspace(0, 1, self.model_length)
        f_values = 1 - x ** 2
        var_values = 0.5 * (1 + np.cos(np.pi * x))

        f_levels = np.digitize(f_values, np.linspace(0, 1, self.observation_levels)) - 1
        var_levels = np.digitize(var_values, np.linspace(0, 1, self.observation_levels)) - 1

        self.fixed_obs_sequence = f_levels * self.observation_levels + var_levels
        self.fixed_obs_sequence = np.clip(self.fixed_obs_sequence, 0, self.n_observations - 1)

    def discretize_observation(self, f_value, var_value):
        """Дискретизация одного наблюдения с защитой от крайних случаев"""
        if len(self.f_best_history) == 0 or len(self.variance_history) == 0:
            return 0

        max_f = max(self.f_best_history) or 1.0
        max_var = max(self.variance_history) or 1.0

        f_norm = min(max(f_value / max_f, 0.0), 1.0)
        var_norm = min(max(var_value / max_var, 0.0), 1.0)

        f_level = np.digitize(f_norm, np.linspace(0, 1, self.observation_levels)) - 1
        var_level = np.digitize(var_norm, np.linspace(0, 1, self.observation_levels)) - 1

        obs = f_level * self.observation_levels + var_level
        return min(int(obs), self.n_observations - 1)

    def update_emissions(self, observations):
        """Обновление матрицы эмиссий с гарантированным увеличением счетчиков"""
        if len(observations) == 0:
            return

        # Получаем последовательность состояний
        states = self.viterbi_algorithm(observations)
        self.state_history.extend(states)

        # Гарантируем увеличение счетчиков
        for s, o in zip(states, observations):
            self.emission_counts[s, o] += 1.0  # Явное увеличение на 1

        # Нормализация
        self.emission_probs = self.emission_counts / self.emission_counts.sum(axis=1, keepdims=True)

    def update_transitions(self, observations=None):
        """Одна итерация обновления матрицы переходов с реальными изменениями"""
        if observations is None:
            observations = self.fixed_obs_sequence

        alpha = self.forward_algorithm(observations)
        beta = self.backward_algorithm(observations)

        # Вычисляем gamma и xi
        gamma = alpha * beta
        gamma /= gamma.sum(axis=1, keepdims=True)

        xi = np.zeros((len(observations) - 1, self.n_states, self.n_states))
        for t in range(len(observations) - 1):
            xi[t] = alpha[t, :, None] * self.transition_probs * \
                    self.emission_probs[:, observations[t + 1]] * beta[t + 1]
            xi[t] /= xi[t].sum()

        # Значительно увеличиваем счетчики для реальных изменений
        for i in range(self.n_states):
            for j in range(self.n_states):
                self.transition_counts[i, j] += np.sum(xi[:, i, j]) * 10  # Усиливаем эффект

        # Обновляем вероятности переходов
        self.transition_probs = self.transition_counts / self.transition_counts.sum(axis=1, keepdims=True)
        self.initial_probs = gamma[0]


    def update_with_optimizer_data(self, f_best_history, variance_history):
        """Основной метод обновления модели"""
        self.f_best_history = f_best_history
        self.variance_history = variance_history

        # Дискретизируем наблюдения
        observations = []
        for f, var in zip(f_best_history, variance_history):
            observations.append(self.discretize_observation(f, var))
        observations = np.array(observations)

        # Обновляем модель
        self.update_emissions(observations)
        self.observation_history.extend(observations.tolist())

    def forward_algorithm(self, observations):
        """Прямой алгоритм"""
        alpha = np.zeros((len(observations), self.n_states))
        alpha[0] = self.initial_probs * self.emission_probs[:, observations[0]]

        for t in range(1, len(observations)):
            for j in range(self.n_states):
                alpha[t, j] = np.sum(alpha[t - 1] * self.transition_probs[:, j]) * \
                              self.emission_probs[j, observations[t]]

        return alpha

    def backward_algorithm(self, observations):
        """Обратный алгоритм"""
        beta = np.zeros((len(observations), self.n_states))
        beta[-1] = 1.0

        for t in range(len(observations) - 2, -1, -1):
            for i in range(self.n_states):
                beta[t, i] = np.sum(
                    self.transition_probs[i, :] *
                    self.emission_probs[:, observations[t + 1]] *
                    beta[t + 1, :]
                )

        return beta

    def viterbi_algorithm(self, observations):
        """Алгоритм Витерби"""
        n_obs = len(observations)
        delta = np.zeros((n_obs, self.n_states))
        psi = np.zeros((n_obs, self.n_states), dtype=int)

        delta[0] = self.initial_probs * self.emission_probs[:, observations[0]]

        for t in range(1, n_obs):
            for j in range(self.n_states):
                trans_probs = delta[t - 1] * self.transition_probs[:, j]
                psi[t, j] = np.argmax(trans_probs)
                delta[t, j] = trans_probs[psi[t, j]] * self.emission_probs[j, observations[t]]

        states = np.zeros(n_obs, dtype=int)
        states[-1] = np.argmax(delta[-1])

        for t in range(n_obs - 2, -1, -1):
            states[t] = psi[t + 1, states[t + 1]]

        return states

    def predict_next_state(self, current_state):
        """Предсказание следующего состояния"""
        probs = self.transition_probs[current_state]
        return np.random.choice(self.n_states, p=probs)

    def get_current_observation(self):
        """Получение текущего дискретизированного наблюдения"""
        if len(self.f_best_history) == 0 or len(self.variance_history) == 0:
            return 0
        return self.discretize_observation(self.f_best_history[-1], self.variance_history[-1])
