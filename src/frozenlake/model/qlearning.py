from . import Model
import numpy as np
from gymnasium import Env

class QLearning(Model):
  def reset(self):
    super().reset()
    self.q = np.zeros((self.state_size, self.action_size), dtype="float32")

  def update(self, state: int, action: int, reward: float, new_state: int | None = None, new_action: int | None = None, final: bool = False):
    target = reward + self.dr * self.q[new_state].max() * (not final)
    self.q[state, action] += self.lr * (target - self.q[state, action])

  def train(self, env: Env, max_iter: int, threshold: float = 1e-9) -> int:
    ep = 0
    while ep < max_iter:
      state, _ = env.reset()
      reward = 0

      done, trunc = False, False
      while not (done or trunc):
        action = self(state, greedy=False)
        new_state, reward, done, trunc, _ = env.step(action)
        self.update(state, action, float(reward), new_state, None, done)
        state = new_state

      if reward != 0: self.er = max(self.er_min, self.er * self.er_decay)
      ep += 1
    return ep

class DoubleQLearning(Model):
  def reset(self):
    super().reset()
    self.qa = np.zeros((self.state_size, self.action_size), dtype="float32")
    self.qb = np.zeros((self.state_size, self.action_size), dtype="float32")

  def __call__(self, state: int, *, greedy: bool = True) -> int:
    return np.random.choice(self.action_size) if not greedy and np.random.uniform() < self.er else self.q[state].argmax().astype("uint8")

  def update(self, state: int, action: int, reward: float, new_state: int | None = None, new_action: int | None = None, final: bool = False):
    if np.random.uniform(0, 1) < 0.5:
      target = reward + self.dr * self.qb[new_state].max() * (not final)
      td_error = target - self.qa[state, action]
      self.qa[state, action] += self.lr * td_error
    else:
      target = reward + self.dr * self.qa[new_state].max() * (not final)
      td_error = target - self.qb[state, action]
      self.qb[state, action] += self.lr * td_error

  def train(self, env: Env, max_iter: int, threshold: float = 1e-9) -> int:
    for _ in range(max_iter):
      state, _ = env.reset()
      term, trunc = False, False
      while not (term or trunc):
        action = self(state, greedy=False)
        new_state, reward, term, trunc, _ = env.step(action)
        self.update(state, action, float(reward), new_state, term)
        state = new_state
      self.er = max(self.er_min, self.er * self.er_decay)

    self.q = (self.qa + self.qb) / 2

    return max_iter