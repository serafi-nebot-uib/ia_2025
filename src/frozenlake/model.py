import numpy as np
from gymnasium import Env

class SARSA:
  def __init__(self, state_size: int, action_size: int, learning_rate: float, gamma: float, eps: float = 1.0, eps_min: float = 0.10, eps_decay: float = 0.995):
    self.state_size, self.action_size = state_size, action_size
    self.learning_rate, self.gamma = learning_rate, gamma
    self.eps, self.eps_min, self.eps_decay = eps, eps_min, eps_decay
    self.reset()

  def reset(self): self.q = np.zeros((self.state_size, self.action_size), dtype="float32")

  def __call__(self, state: int, training: bool = False):
    if training and np.random.uniform(0, 1) < self.eps: return np.random.choice(self.action_size)
    else: return self.q[state].argmax()

  def update(self, state: int, action: int, reward: float, new_state: int, new_action: int, final: bool = False):
    target = reward + self.gamma * self.q[new_state, new_action] * (not final)
    self.q[state, action] += self.learning_rate * (target - self.q[state, action])

  def train(self, env: Env, episodes: int):
    for _ in range(episodes):
      state, _ = env.reset()
      action = self(state)
      term, trunc = False, False
      while not (term or trunc):
        new_state, reward, term, trunc, _ = env.step(action)
        new_action = self(new_state, True)
        self.update(state, action, reward, new_state, new_action, term)
        state, action = new_state, new_action
      self.eps = max(self.eps_min, self.eps * self.eps_decay)

class QLearning:
  def __init__(self, state_size: int, action_size: int, learning_rate: float, gamma: float, eps: float = 1.0, eps_min: float = 0.10, eps_decay: float = 0.995):
    self.state_size, self.action_size = state_size, action_size
    self.learning_rate, self.gamma = learning_rate, gamma
    self.eps, self.eps_min, self.eps_decay = eps, eps_min, eps_decay
    self.reset()

  def reset(self): self.q = np.zeros((self.state_size, self.action_size), dtype="float32")

  def __call__(self, state: int, training: bool = False):
    if training and np.random.uniform(0, 1) < self.eps: return np.random.choice(self.action_size)
    else: return self.q[state].argmax()

  def update(self, state: int, action: int, reward: float, new_state: int, final: bool = False):
    target = reward + self.gamma * self.q[new_state].max() * (not final)
    self.q[state, action] += self.learning_rate * (target - self.q[state, action])

  def train(self, env: Env, episodes: int):
    for _ in range(episodes):
      state, _ = env.reset()
      term, trunc = False, False
      while not (term or trunc):
        action = self(state, True)
        new_state, reward, term, trunc, _ = env.step(action)
        self.update(state, action, reward, new_state, term)
        state = new_state
      self.eps = max(self.eps_min, self.eps * self.eps_decay)

class DoubleQLearning:
  def __init__(self, state_size: int, action_size: int, learning_rate: float, gamma: float, eps: float = 1.0, eps_min: float = 0.10, eps_decay: float = 0.995):
    self.state_size, self.action_size = state_size, action_size
    self.learning_rate, self.gamma = learning_rate, gamma
    self.eps, self.eps_min, self.eps_decay = eps, eps_min, eps_decay
    self.reset()

  def reset(self):
    self.qa = np.zeros((self.state_size, self.action_size), dtype="float32")
    self.qb = np.zeros((self.state_size, self.action_size), dtype="float32")

  def __call__(self, state: int, training: bool = False):
    if training and np.random.uniform(0, 1) < self.eps: return np.random.choice(self.action_size)
    else: return np.argmax((self.qa[state] + self.qb[state]) / 2)

  def update(self, state: int, action: int, reward: float, new_state: int, final: bool = False):
    if np.random.uniform(0, 1) < 0.5:
      target = reward + self.gamma * self.qb[new_state].max() * (not final)
      td_error = target - self.qa[state, action]
      self.qa[state, action] += self.learning_rate * td_error
    else:
      target = reward + self.gamma * self.qa[new_state].max() * (not final)
      td_error = target - self.qb[state, action]
      self.qb[state, action] += self.learning_rate * td_error

  def train(self, env: Env, episodes: int):
    for _ in range(episodes):
      state, _ = env.reset()
      term, trunc = False, False
      while not (term or trunc):
        action = self(state, True)
        new_state, reward, term, trunc, _ = env.step(action)
        self.update(state, action, reward, new_state, term)
        state = new_state
      self.eps = max(self.eps_min, self.eps * self.eps_decay)