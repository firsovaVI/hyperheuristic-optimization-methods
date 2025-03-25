import numpy as np


class HiddenMarkovModel:
    def __init__(self, n_states, n_observations):
        self.n_states = n_states
        self.n_observations = n_observations
        self._init_balanced_parameters()

    def _init_balanced_parameters(self):
        # Равные начальные вероятности
        self.transition_probs = np.full((self.n_states, self.n_states), 0.5)
        self.emission_probs = np.full((self.n_states, self.n_observations), 0.5)
        self.initial_probs = np.array([0.5, 0.5])

        # Небольшой шум для打破 симметрии
        self.transition_probs += np.random.uniform(-0.1, 0.1, size=(self.n_states, self.n_states))
        self.transition_probs = np.clip(self.transition_probs, 0.1, 0.9)
        self.transition_probs /= self.transition_probs.sum(axis=1, keepdims=True)

    def predict_next_state(self, current_state):
        return np.argmax(self.transition_probs[current_state])

    def update_model(self, current_state, observation):
        # Плавное обновление с learning_rate=0.1
        learning_rate = 0.1
        self.transition_probs[current_state] *= (1 - learning_rate)
        self.transition_probs[current_state] += learning_rate * observation

        self.emission_probs[current_state, observation] *= (1 - learning_rate)
        self.emission_probs[current_state, observation] += learning_rate * 0.9

        # Нормализация
        self.transition_probs[current_state] /= self.transition_probs[current_state].sum()
        self.emission_probs[current_state] /= self.emission_probs[current_state].sum()
