import numpy as np


class HiddenMarkovModel:
    def __init__(self, n_states, n_observations):
        """
        Инициализация скрытой марковской модели со случайными начальными параметрами.

        Args:
            n_states (int): Количество скрытых состояний.
            n_observations (int): Количество наблюдаемых состояний.
        """
        self.n_states = n_states
        self.n_observations = n_observations
        self._init_random_parameters()

    def _init_random_parameters(self):
        """
        Генерирует случайные начальные параметры модели.
        """
        # Случайные вероятности переходов
        self.transition_probs = np.random.rand(self.n_states, self.n_states)
        self.transition_probs /= self.transition_probs.sum(axis=1, keepdims=True)  # Нормализация

        # Случайные вероятности излучения
        self.emission_probs = np.random.rand(self.n_states, self.n_observations)
        self.emission_probs /= self.emission_probs.sum(axis=1, keepdims=True)  # Нормализация

        # Случайные начальные вероятности
        self.initial_probs = np.random.rand(self.n_states)
        self.initial_probs /= self.initial_probs.sum()  # Нормализация

    def _forward(self, observations):
        """
        Реализация прямого алгоритма (алгоритм вперед).

          Args:
              observations (np.array): Последовательность наблюдаемых состояний.

          Returns:
              alpha (np.array): Матрица альфа.
        """
        T = len(observations)
        alpha = np.zeros((T, self.n_states))

        # Инициализация alpha для t=0
        for s in range(self.n_states):
            alpha[0, s] = self.initial_probs[s] * self.emission_probs[s, observations[0]]

        # Рекурсия по времени
        for t in range(1, T):
            for s in range(self.n_states):
                for s_prev in range(self.n_states):
                    alpha[t, s] += alpha[t - 1, s_prev] * self.transition_probs[s_prev, s]
                alpha[t, s] *= self.emission_probs[s, observations[t]]
        return alpha

    def _backward(self, observations):
        """
         Реализация обратного алгоритма (алгоритма назад).

          Args:
              observations (np.array): Последовательность наблюдаемых состояний.

          Returns:
              beta (np.array): Матрица бета.
        """
        T = len(observations)
        beta = np.zeros((T, self.n_states))

        # Инициализация beta для t=T-1
        for s in range(self.n_states):
            beta[T - 1, s] = 1

        # Рекурсия по времени
        for t in range(T - 2, -1, -1):
            for s in range(self.n_states):
                for s_next in range(self.n_states):
                    beta[t, s] += beta[t + 1, s_next] * self.transition_probs[s, s_next] * self.emission_probs[
                        s_next, observations[t + 1]]
        return beta

    def _viterbi(self, observations):
        """
        Реализация алгоритма Витерби для декодирования наиболее вероятной последовательности скрытых состояний.

        Args:
            observations (np.array): Последовательность наблюдаемых состояний.

        Returns:
            best_path (np.array): Наиболее вероятная последовательность скрытых состояний.
        """
        T = len(observations)
        viterbi = np.zeros((T, self.n_states))
        backpointer = np.zeros((T, self.n_states), dtype=int)

        # Инициализация для t=0
        for s in range(self.n_states):
            viterbi[0, s] = self.initial_probs[s] * self.emission_probs[s, observations[0]]

        # Рекурсия по времени
        for t in range(1, T):
            for s in range(self.n_states):
                max_prob = 0
                max_state = 0
                for s_prev in range(self.n_states):
                    prob = viterbi[t - 1, s_prev] * self.transition_probs[s_prev, s]
                    if prob > max_prob:
                        max_prob = prob
                        max_state = s_prev
                viterbi[t, s] = max_prob * self.emission_probs[s, observations[t]]
                backpointer[t, s] = max_state

        # Поиск наиболее вероятного последнего состояния
        best_last_state = np.argmax(viterbi[T - 1, :])
        best_path = np.zeros(T, dtype=int)
        best_path[T - 1] = best_last_state
        for t in range(T - 2, -1, -1):
            best_path[t] = backpointer[t + 1, best_path[t + 1]]
        return best_path

    def _log_likelihood(self, observations):
        """
        Вычисляет логарифм правдоподобия последовательности наблюдений.
        Args:
            observations (np.array): последовательность наблюдаемых состояний

        Returns:
           (float): Логарифм правдоподобия.
        """
        alpha = self._forward(observations)
        return np.log(np.sum(alpha[-1]))

    def baum_welch(self, observations, max_iterations=100, tolerance=1e-5):
        """
          Обучает модель по последовательности наблюдений, используя алгоритм Баума-Велша.
          Args:
            observations (np.array): последовательность наблюдений
            max_iterations (int): максимальное количество итераций обучения
            tolerance (float): уровень допуска для сходимости
          Returns:
            log_likelihood (float): Логарифм правдоподобия.
        """
        observations = np.asarray(observations, dtype=int)
        old_log_likelihood = float('-inf')

        for iteration in range(max_iterations):
            alpha = self._forward(observations)
            beta = self._backward(observations)

            T = len(observations)
            xi = np.zeros((T - 1, self.n_states, self.n_states))
            gamma = np.zeros((T, self.n_states))

            for t in range(T - 1):
                denominator = 0
                for i in range(self.n_states):
                    for j in range(self.n_states):
                        denominator += alpha[t, i] * self.transition_probs[i, j] * self.emission_probs[
                            j, observations[t + 1]] * beta[t + 1, j]

                for i in range(self.n_states):
                    for j in range(self.n_states):
                        numerator = alpha[t, i] * self.transition_probs[i, j] * self.emission_probs[
                            j, observations[t + 1]] * beta[t + 1, j]
                        xi[t, i, j] = numerator / denominator

            for t in range(T):
                gamma[t] = np.sum(xi[t, :, :], axis=1) if t < T - 1 else np.sum(xi[T - 2, :, :],
                                                                                axis=0)  # для последнего элемента t = T-1, но xi[T-1] отсутствует.

            # Обновление начальных вероятностей
            self.initial_probs = gamma[0] / np.sum(gamma[0])

            # Обновление матрицы переходов
            for i in range(self.n_states):
                for j in range(self.n_states):
                    numerator = np.sum(xi[:, i, j])
                    denominator = np.sum(gamma[:-1, i])
                    self.transition_probs[i, j] = numerator / denominator if denominator != 0 else 0

            # Обновление матрицы излучения
            for i in range(self.n_states):
                for k in range(self.n_observations):
                    numerator = np.sum(gamma[observations == k, i])
                    denominator = np.sum(gamma[:, i])
                    self.emission_probs[i, k] = numerator / denominator if denominator != 0 else 0

            new_log_likelihood = self._log_likelihood(observations)
            if abs(new_log_likelihood - old_log_likelihood) < tolerance:
                print("Converged at iteration:", iteration)
                break
            if iteration == max_iterations - 1:
                print("Max iteration reached")
            old_log_likelihood = new_log_likelihood
        return new_log_likelihood


if __name__ == '__main__':
    # Пример использования:
    n_states = 2  # 0 - волки, 1 - лисы
    n_observations = 2  # 0 - зайцы, 1 - лани

    hmm_model = HiddenMarkovModel(n_states, n_observations)

    # Генерируем последовательность
    sequence_length = 20
    # hidden_states, observations = hmm_model.generate_sequence(sequence_length)
    observations = np.random.randint(0, n_observations, sequence_length)

    # Названия
    hidden_state_names = ["Волк", "Лиса"]
    observation_names = ["Заяц", "Лань"]

    # Выводим результат
    # print("Скрытые состояния (хищники):", [hidden_state_names[state] for state in hidden_states])
    print("Наблюдаемые состояния (добыча):", [observation_names[obs] for obs in observations])

    decoded_hidden_states = hmm_model._viterbi(observations)
    print("Декодированные скрытые состояния (хищники):", [hidden_state_names[state] for state in decoded_hidden_states])

    new_observations = np.array([0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0])
    hmm_model.baum_welch(new_observations)
    new_decoded_states = hmm_model._viterbi(new_observations)
    print("Новая последовательность декодированных состояний",
          [hidden_state_names[state] for state in new_decoded_states])
