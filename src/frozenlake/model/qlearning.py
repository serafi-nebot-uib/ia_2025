import time
import numpy as np
from gymnasium import Env
from frozenlake.model import Model

class QLearning(Model):
  def __init__(self, state_size: int, action_size: int, lr: float, dr: float, er: float, er_min: float, er_decay: float):
    super().__init__(state_size, action_size)
    self.name = "QL"
    self.lr, self.dr = lr, dr
    self.er, self.er_min, self.er_decay = er, er_min, er_decay
    self.q = np.zeros((self.state_size, self.action_size), dtype="float32")

  def action(self, state: int, greedy: bool = True) -> int:
    if not greedy and np.random.uniform() < self.er: return np.random.choice(self.action_size)
    else: return np.random.choice(np.flatnonzero(self.q[state] == self.q[state].max()))

  def update(self, state: int, action: int, reward: float, new_state: int, final: bool):
    target = reward + self.dr * self.q[new_state].max() * (not final)
    self.q[state, action] += self.lr * (target - self.q[state, action])

  def train(self, env: Env, max_iter: int) -> int:
    iter = 0
    while iter < max_iter:
      iter_start = time.perf_counter()
      state, _ = env.reset()
      reward, reward_total = 0.0, 0.0

      done, trunc = False, False
      while not (done or trunc):
        action = self.action(state, greedy=False)
        new_state, reward, done, trunc, _ = env.step(action)
        reward = float(reward)
        self.update(state, action, reward, new_state, done)
        state = new_state
        reward_total += reward

      self.er = max(self.er_min, self.er * self.er_decay)
      iter += 1
      iter_end = time.perf_counter()
      self.train_stats(iter, reward_total, iter_end - iter_start)

    return iter

class DoubleQLearning(Model):
  def __init__(self, state_size: int, action_size: int, lr: float, dr: float, er: float, er_min: float, er_decay: float):
    super().__init__(state_size, action_size)
    self.name = "2QL"
    self.lr, self.dr = lr, dr
    self.lr, self.dr = lr, dr
    self.er, self.er_min, self.er_decay = er, er_min, er_decay
    self.qa = np.zeros((self.state_size, self.action_size), dtype="float32")
    self.qb = np.zeros((self.state_size, self.action_size), dtype="float32")

  def action(self, state: int, greedy: bool = True) -> int:
    if not greedy and np.random.uniform() < self.er: 
      return np.random.choice(self.action_size)
    else:
      q = (self.qa[state] + self.qb[state]) / 2
      return np.random.choice(np.flatnonzero(q == q.max()))

  def update(self, state: int, action: int, reward: float, new_state: int, final: bool):
    qa, qb = (self.qa, self.qb) if np.random.uniform() < 0.50 else (self.qb, self.qa)
    target = reward + self.dr * qb[new_state].max() * (not final)
    qa[state, action] += self.lr * (target - qa[state, action])

  def train(self, env: Env, max_iter: int) -> int:
    iter = 0
    while iter < max_iter:
      iter_start = time.perf_counter()
      state, _ = env.reset()
      reward, reward_total = 0.0, 0.0

      done, trunc = False, False
      while not (done or trunc):
        action = self.action(state, greedy=False)
        new_state, reward, done, trunc, _ = env.step(action)
        reward = float(reward)
        self.update(state, action, reward, new_state, done)
        state = new_state
        reward_total += reward

      self.er = max(self.er_min, self.er * self.er_decay)
      iter += 1
      iter_end = time.perf_counter()
      self.train_stats(iter, reward_total, iter_end - iter_start)

    return iter