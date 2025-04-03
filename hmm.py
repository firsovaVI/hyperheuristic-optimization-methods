import numpy as np
from scipy.stats import norm


class HiddenMarkovModel:
    def __init__(self, n_states, n_observations, model_length=16):
        """
        Инициализация скрытой марковской модели

        Параметры:
            n_states: количество скрытых состояний
            n_observations: количество возможных наблюдений
            model_length: длина последовательности для обучения
        """
        self.n_states = n_states
        self.n_observations = n_observations
        self.model_length = model_length
        self.observation_levels = 8  # Уровни дискретизации

        # Инициализация параметров модели
        self.transition_probs = np.ones((n_states, n_states)) / n_states
        self.initial_probs = np.ones(n_states) / n_states
        self.emission_probs = np.ones((n_states, n_observations)) / n_observations

        # Сразу устанавливаем фиксированную последовательность с плавным уменьшением
        self._initialize_smooth_sequence()

        # История наблюдений
        self.observation_history = []

    def _initialize_smooth_sequence(self):
        """Инициализация плавной последовательности для Baum-Welch"""
        x = np.linspace(0, 1, self.model_length)

        # Параболическое уменьшение значений функции (1 -> 0)
        f_values = 1 - x ** 2

        # Косинусное уменьшение дисперсии (1 -> 0)
        var_values = 0.5 * (1 + np.cos(np.pi * x))

        # Дискретизация на 8 уровней
        f_levels = np.digitize(f_values, np.linspace(0, 1, self.observation_levels)) - 1
        var_levels = np.digitize(var_values, np.linspace(0, 1, self.observation_levels)) - 1

        # Комбинация в наблюдения (0-63)
        self.fixed_obs_sequence = f_levels * self.observation_levels + var_levels
        self.fixed_obs_sequence = np.clip(self.fixed_obs_sequence, 0, self.n_observations - 1)

    def set_custom_fixed_sequence(self, f_values, var_values):
        """
        Установка пользовательской фиксированной последовательности

        Параметры:
            f_values: массив значений функции (длина model_length)
            var_values: массив значений дисперсии (длина model_length)
        """
        if len(f_values) != self.model_length or len(var_values) != self.model_length:
            raise ValueError(f"Длина последовательностей должна быть {self.model_length}")

        # Нормализация и дискретизация
        f_norm = np.array(f_values) / (np.max(f_values) + 1e-10)
        var_norm = np.array(var_values) / (np.max(var_values) + 1e-10)

        f_levels = np.digitize(f_norm, np.linspace(0, 1, self.observation_levels)) - 1
        var_levels = np.digitize(var_norm, np.linspace(0, 1, self.observation_levels)) - 1

        self.fixed_obs_sequence = f_levels * self.observation_levels + var_levels
        self.fixed_obs_sequence = np.clip(self.fixed_obs_sequence, 0, self.n_observations - 1)

    def discretize_optimizer_results(self, f_best_history, variance_history):
        """Дискретизация данных оптимизации и обновление emission_probs"""
        # Нормализация
        f_norm = np.array(f_best_history) / (np.max(f_best_history) + 1e-10)
        s_norm = np.array(variance_history) / (np.max(variance_history) + 1e-10)

        # Дискретизация
        f_levels = np.digitize(f_norm, np.linspace(0, 1, self.observation_levels)) - 1
        s_levels = np.digitize(s_norm, np.linspace(0, 1, self.observation_levels)) - 1

        observations = f_levels * self.observation_levels + s_levels
        observations = np.clip(observations, 0, self.n_observations - 1)

        # Обновление матрицы эмиссий
        self._update_emission_probs(observations)

        return observations

    def _update_emission_probs(self, observations):
        """Обновление матрицы эмиссий"""
        counts = np.ones((self.n_states, self.n_observations))  # Сглаживание

        if hasattr(self, 'state_sequence') and len(self.state_sequence) >= len(observations):
            states = self.state_sequence[-len(observations):]
            for s, o in zip(states, observations):
                counts[s, o] += 1
        else:
            for o in observations:
                counts[:, o] += 1.0 / self.n_states

        self.emission_probs = counts / counts.sum(axis=1, keepdims=True)

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

    def baum_welch(self, observations):
        """Алгоритм Баума-Велча (только для transition_probs)"""
        alpha = self.forward_algorithm(observations)
        beta = self.backward_algorithm(observations)

        gamma = alpha * beta
        gamma /= gamma.sum(axis=1, keepdims=True)

        xi = np.zeros((len(observations) - 1, self.n_states, self.n_states))
        for t in range(len(observations) - 1):
            xi[t] = alpha[t, :, None] * self.transition_probs * \
                    self.emission_probs[:, observations[t + 1]] * beta[t + 1]
            xi[t] /= xi[t].sum()

        self.initial_probs = gamma[0]
        for i in range(self.n_states):
            for j in range(self.n_states):
                self.transition_probs[i, j] = np.sum(xi[:, i, j]) / np.sum(gamma[:-1, i])

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

    def update_with_optimizer_data(self, f_best_history, variance_history):
        """Основной метод обновления модели"""
        observations = self.discretize_optimizer_results(f_best_history, variance_history)
        self.observation_history.extend(observations.tolist())
        self.baum_welch(self.fixed_obs_sequence)  # Используем фиксированную последовательность

    def predict_next_state(self, current_state):
        """Предсказание следующего состояния"""
        probs = self.transition_probs[current_state]
        return np.random.choice(self.n_states, p=probs)
