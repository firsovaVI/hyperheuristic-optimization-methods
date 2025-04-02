import numpy as np
from scipy.stats import norm


class HiddenMarkovModel:
    def __init__(self, n_states, n_observations, model_length=16):
        """
        Инициализация скрытой марковской модели

        Параметры:
        n_states - количество скрытых состояний
        n_observations - количество возможных наблюдений
        model_length - длина последовательности для обучения
        """
        self.n_states = n_states
        self.n_observations = n_observations
        self.model_length = model_length
        self._init_balanced_parameters()
        self.observation_levels = 8  # Уровни дискретизации для наблюдений
        self.state_sequence = []  # Последовательность состояний
        self.observation_sequence = []  # Последовательность наблюдений

    def _init_balanced_parameters(self):
        """Инициализация параметров модели сбалансированными значениями"""
        # Матрица переходов между состояниями
        self.transition_probs = np.random.dirichlet(np.ones(self.n_states), size=self.n_states)

        # Матрица эмиссии (вероятности наблюдений для каждого состояния)
        self.emission_probs = np.random.dirichlet(np.ones(self.n_observations), size=self.n_states)

        # Начальные вероятности состояний
        self.initial_probs = np.random.dirichlet(np.ones(self.n_states))

    def forward_algorithm(self, observations):
        """Прямой алгоритм (вычисление альфа-вероятностей)"""
        alpha = np.zeros((len(observations), self.n_states))
        log_alpha = np.zeros_like(alpha)

        # Инициализация с логарифмами для численной стабильности
        log_alpha[0] = np.log(self.initial_probs + 1e-100) + \
                       np.log(self.emission_probs[:, observations[0]] + 1e-100)

        for t in range(1, len(observations)):
            for j in range(self.n_states):
                log_alpha[t, j] = np.logaddexp.reduce(
                    log_alpha[t - 1] + np.log(self.transition_probs[:, j] + 1e-100)
                ) + np.log(self.emission_probs[j, observations[t]] + 1e-100)

        return np.exp(log_alpha)

    def backward_algorithm(self, observations):
        """Обратный алгоритм (вычисление бета-вероятностей)"""
        beta = np.zeros((len(observations), self.n_states))
        log_beta = np.zeros_like(beta)

        # Инициализация
        log_beta[-1] = 0.0  # log(1) = 0

        for t in range(len(observations) - 2, -1, -1):
            for i in range(self.n_states):
                log_beta[t, i] = np.logaddexp.reduce(
                    log_beta[t + 1] +
                    np.log(self.transition_probs[i, :] + 1e-100) +
                    np.log(self.emission_probs[:, observations[t + 1]] + 1e-100)
                )

        return np.exp(log_beta)

    def baum_welch(self, observations, max_iter=100, tol=1e-6):
        """Алгоритм Баума-Велча для обучения модели"""
        log_likelihood_prev = -np.inf

        for iteration in range(max_iter):
            # E-шаг
            alpha = self.forward_algorithm(observations)
            beta = self.backward_algorithm(observations)

            # Защита от нулевых значений
            alpha = np.clip(alpha, 1e-100, None)
            beta = np.clip(beta, 1e-100, None)

            xi = np.zeros((len(observations) - 1, self.n_states, self.n_states))
            gamma = np.zeros((len(observations), self.n_states))

            for t in range(len(observations) - 1):
                denominator = np.sum(alpha[t] * beta[t])
                if denominator == 0:
                    denominator = 1e-100

                for i in range(self.n_states):
                    gamma[t, i] = alpha[t, i] * beta[t, i] / denominator

                    for j in range(self.n_states):
                        xi[t, i, j] = alpha[t, i] * self.transition_probs[i, j] * \
                                      self.emission_probs[j, observations[t + 1]] * beta[t + 1, j] / denominator

            # Последний gamma
            denominator = np.sum(alpha[-1] * beta[-1])
            if denominator == 0:
                denominator = 1e-100
            gamma[-1] = alpha[-1] * beta[-1] / denominator

            # M-шаг
            self.initial_probs = gamma[0] / np.sum(gamma[0])

            # Обновление матрицы переходов
            for i in range(self.n_states):
                denominator = np.sum(gamma[:-1, i])
                if denominator == 0:
                    denominator = 1e-100
                for j in range(self.n_states):
                    self.transition_probs[i, j] = np.sum(xi[:, i, j]) / denominator

            # Обновление матрицы эмиссии
            for j in range(self.n_states):
                denominator = np.sum(gamma[:, j])
                if denominator == 0:
                    denominator = 1e-100
                for k in range(self.n_observations):
                    mask = (observations == k)
                    self.emission_probs[j, k] = np.sum(gamma[mask, j]) / denominator

            # Проверка сходимости
            current_log_likelihood = np.log(np.sum(alpha[-1])) if np.sum(alpha[-1]) > 0 else -np.inf
            if abs(current_log_likelihood - log_likelihood_prev) < tol:
                break
            log_likelihood_prev = current_log_likelihood

    def viterbi_algorithm(self, observations):
        """Алгоритм Витерби для нахождения наиболее вероятной последовательности состояний"""
        n_obs = len(observations)
        delta = np.zeros((n_obs, self.n_states))
        psi = np.zeros((n_obs, self.n_states), dtype=int)

        # Инициализация
        delta[0] = self.initial_probs * self.emission_probs[:, observations[0]]

        # Рекурсия
        for t in range(1, n_obs):
            for j in range(self.n_states):
                trans_probs = delta[t - 1] * self.transition_probs[:, j]
                psi[t, j] = np.argmax(trans_probs)
                delta[t, j] = trans_probs[psi[t, j]] * self.emission_probs[j, observations[t]]

        # Обратный ход
        states = np.zeros(n_obs, dtype=int)
        states[-1] = np.argmax(delta[-1])

        for t in range(n_obs - 2, -1, -1):
            states[t] = psi[t + 1, states[t + 1]]

        return states

    def predict_next_state(self, current_state):
        """Вероятностное предсказание следующего состояния"""
        r = np.random.random()
        cumulative = 0.0
        for j in range(self.n_states):
            cumulative += self.transition_probs[current_state, j]
            if r <= cumulative:
                return j
        return self.n_states - 1

    def discretize_optimizer_results(self, f_best_history, variance_history):
        """Дискретизация результатов оптимизации для использования в HMM"""
        # Нормализация
        f_normalized = f_best_history / (np.max(f_best_history) + 1e-100)
        s_normalized = variance_history / (np.max(variance_history) + 1e-100)

        # Дискретизация на 8 уровней (0-7)
        f_levels = np.digitize(f_normalized, np.linspace(0, 1, self.observation_levels)) - 1
        s_levels = np.digitize(s_normalized, np.linspace(0, 1, self.observation_levels)) - 1

        # Комбинация в одно наблюдение (0-63)
        observations = f_levels * self.observation_levels + s_levels
        return np.clip(observations, 0, self.n_observations - 1)

    def update_with_optimizer_data(self, f_best_history, variance_history):
        """Обновление модели на основе данных оптимизатора"""
        observations = self.discretize_optimizer_results(f_best_history, variance_history)
        self.observation_sequence.extend(observations.tolist())

        # Ограничение длины последовательности
        if len(self.observation_sequence) > self.model_length:
            self.observation_sequence = self.observation_sequence[-self.model_length:]

        # Обучение модели при наличии достаточных данных
        if len(self.observation_sequence) == self.model_length:
            self.baum_welch(np.array(self.observation_sequence))
