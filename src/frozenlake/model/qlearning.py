import time
import numpy as np
from gymnasium import Env
from frozenlake.model import Model
from frozenlake.const import DEBUG, STATS, SAMPLE

class QLearning(Model):
  def __init__(self, state_size: int, action_size: int, lr: float, dr: float, er: float, er_min: float, er_decay: float):
    super().__init__(state_size, action_size)
    self.lr, self.dr = lr, dr
    self.er, self.er_min, self.er_decay = er, er_min, er_decay
    self.q = np.zeros((self.state_size, self.action_size), dtype="float32")

  def action(self, state: int, greedy: bool = True) -> int:
    return np.random.choice(self.action_size) if not greedy and np.random.uniform() < self.er else self.q[state].argmax()

  def update(self, state: int, action: int, reward: float, new_state: int, final: bool):
    target = reward + self.dr * self.q[new_state].max() * (not final)
    self.q[state, action] += self.lr * (target - self.q[state, action])

  def train(self, env: Env, max_iter: int, threshold: float = 1e-9) -> int:
    iter = 0
    iter_start = time.perf_counter()
    while iter < max_iter:
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

      if reward_total > 0: self.er = max(self.er_min, self.er * self.er_decay)
      iter += 1

      if DEBUG > 0 or STATS > 0: self.rewards.append(reward_total)
      if iter % SAMPLE == 0 and DEBUG > 0:
        iter_end = time.perf_counter()
        avg = np.mean(self.rewards[-SAMPLE:])
        print(f"{iter:>7d} | {avg:>4.2f} | {iter_end - iter_start:.6f} sec")
        iter_start = time.perf_counter()

    return iter

class DoubleQLearning(Model):
  def __init__(self, state_size: int, action_size: int, lr: float, dr: float, er: float, er_min: float, er_decay: float):
    super().__init__(state_size, action_size)
    self.lr, self.dr = lr, dr
    self.er, self.er_min, self.er_decay = er, er_min, er_decay
    self.qa = np.zeros((self.state_size, self.action_size), dtype="float32")
    self.qb = np.zeros((self.state_size, self.action_size), dtype="float32")

  def action(self, state: int, greedy: bool = True) -> int:
    if not greedy and np.random.uniform() < self.er: return np.random.choice(self.action_size)
    else: return int(np.argmax((self.qa[state] + self.qb[state]) / 2))

  def update(self, state: int, action: int, reward: float, new_state: int, final: bool):
    qa, qb = (self.qa, self.qb) if np.random.uniform() < 0.50 else (self.qb, self.qa)
    target = reward + self.dr * qb[new_state].max() * (not final)
    qa[state, action] += self.lr * (target - qa[state, action])

  def train(self, env: Env, max_iter: int, threshold: float = 1e-9) -> int:
    iter = 0
    iter_start = time.perf_counter()
    while iter < max_iter:
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

      if reward_total > 0: self.er = max(self.er_min, self.er * self.er_decay)
      iter += 1

      if DEBUG > 0 or STATS > 0: self.rewards.append(reward_total)
      if iter % SAMPLE == 0 and DEBUG > 0:
        iter_end = time.perf_counter()
        avg = np.mean(self.rewards[-SAMPLE:])
        print(f"{iter:>7d} | {avg:>4.2f} | {iter_end - iter_start:.6f} sec")
        iter_start = time.perf_counter()

    return iter