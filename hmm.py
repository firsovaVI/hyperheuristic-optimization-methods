import numpy as np


class HiddenMarkovModel:
    def __init__(self, n_states, n_observations):
        self.n_states = n_states
        self.n_observations = n_observations
        self._init_balanced_parameters()

    def _init_balanced_parameters(self):
        # Инициализация матрицы переходов (n_states x n_states)
        self.transition_probs = np.random.dirichlet(np.ones(self.n_states), size=self.n_states)

        # Инициализация матрицы эмиссии (n_states x n_observations)
        self.emission_probs = np.random.dirichlet(np.ones(self.n_observations), size=self.n_states)

        # Инициализация начальных вероятностей
        self.initial_probs = np.random.dirichlet(np.ones(self.n_states))

    def forward_algorithm(self, observations):
        alpha = np.zeros((len(observations), self.n_states))
        alpha[0] = self.initial_probs * self.emission_probs[:, observations[0]]

        for t in range(1, len(observations)):
            for j in range(self.n_states):
                alpha[t, j] = np.sum(alpha[t - 1] * self.transition_probs[:, j]) * \
                              self.emission_probs[j, observations[t]]

        return np.sum(alpha[-1])

    def viterbi_algorithm(self, observations):
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

    def update_model(self, current_state, observation):
        learning_rate = 0.1
        self.transition_probs[current_state] *= (1 - learning_rate)
        self.transition_probs[current_state] += learning_rate * observation
        self.emission_probs[current_state, observation] *= (1 - learning_rate)
        self.emission_probs[current_state, observation] += learning_rate * 0.9

        self.transition_probs[current_state] /= self.transition_probs[current_state].sum()
        self.emission_probs[current_state] /= self.emission_probs[current_state].sum()

    def predict_next_state(self, current_state):
        return np.argmax(self.transition_probs[current_state])
