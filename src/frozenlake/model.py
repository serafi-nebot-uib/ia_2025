import numpy as np
from gymnasium import Env

class SARSA:
  def __init__(self, state_size: int, action_size: int, learning_rate: float, gamma: float, epsilon: float):
    self.state_size, self.action_size = state_size, action_size
    self.learning_rate, self.gamma, self.epsilon = learning_rate, gamma, epsilon
    self.reset()

  def reset(self): self.q = np.zeros((self.state_size, self.action_size), dtype="float")

  def __call__(self, state: int):
    if np.random.uniform(0, 1) < self.epsilon: return np.random.choice(self.action_size)
    else: return np.random.choice(np.where(self.q[state] == self.q[state].max())[0])

  def update(self, state: int, action: int, reward: float, new_state: int, new_action: int, final: bool = False):
    target = reward + self.gamma * self.q[new_state, new_action] * (not final)
    self.q[state, action] += self.learning_rate * (target - self.q[state, action])

  # TODO: should env be outside of model?
  def train(self, env: Env, episodes: int):
    for _ in range(episodes):
      state, _ = env.reset()
      action = self(state)
      term, trunc = False, False
      while not (term or trunc):
        new_state, reward, term, trunc, _ = env.step(action)
        new_action = self(new_state)
        self.update(state, action, reward, new_state, new_action, term)
        state, action = new_state, new_action

class QLearning:
  def __init__(self, state_size: int, action_size: int, learning_rate: float, gamma: float, epsilon: float):
    self.state_size, self.action_size = state_size, action_size
    self.learning_rate, self.gamma, self.epsilon = learning_rate, gamma, epsilon
    self.reset()

  def reset(self): self.q = np.zeros((self.state_size, self.action_size), dtype="float")

  def __call__(self, state: int):
    if np.random.uniform(0, 1) < self.epsilon: return np.random.choice(self.action_size)
    else: return np.random.choice(np.where(self.q[state] == self.q[state].max())[0])

  def update(self, state: int, action: int, reward: float, new_state: int, final: bool = False):
    target = reward + self.gamma * self.q[new_state].max() * (not final)
    self.q[state, action] += self.learning_rate * (target - self.q[state, action])

  # TODO: should env be outside of model?
  def train(self, env: Env, episodes: int):
    for _ in range(episodes):
      state, _ = env.reset()
      term, trunc = False, False
      while not (term or trunc):
        action = self(state)
        new_state, reward, term, trunc, _ = env.step(action)
        self.update(state, action, reward, new_state, term)
        state = new_state