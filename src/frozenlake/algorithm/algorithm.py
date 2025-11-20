import numpy as np
from gymnasium import Env
from frozenlake.const import DEBUG, STATS, SAMPLE

class Algorithm:
  def __init__(self, state_size: int, action_size: int):
    self.name = self.__class__.__name__
    self.state_size, self.action_size = state_size, action_size
    self.time, self.performance = [], []

  def action(self, state: int, greedy: bool = True) -> int: raise NotImplementedError()
  def train(self, env: Env, max_iter: int) -> int: raise NotImplementedError()

  def train_stats(self, iter: int, reward: float, t: float, sample: int = SAMPLE):
    if DEBUG > 0 or STATS > 0:
      self.performance.append(reward)
      self.time.append(t)

    if DEBUG > 0 and (not sample or iter % sample == 0):
      perf_avg = np.mean(self.performance[-sample:])
      time_avg = np.mean(self.time[-sample:])
      time_total = np.sum(self.time[-sample:])
      print(f"{self.name:<15} {iter:>7d} | perf: {perf_avg:>4.2f} | t: {time_total:.6f} s (total) {time_avg:.6f} s (avg)")

  def test(self, env: Env, num_iter: int = 1) -> float:
    s = 0.0
    for _ in range(num_iter):
      state, _ = env.reset()
      done, trunc, reward = False, False, 0
      while not (done or trunc): state, reward, done, trunc, _ = env.step(self.action(state, greedy=True))
      s += float(reward)
    return s / num_iter

class LearningAlgorithm(Algorithm):
  def __init__(self, state_size: int, action_size: int, lr: float, dr: float, er: float, er_min: float, er_decay: float):
    super().__init__(state_size, action_size)
    self.lr, self.dr = lr, dr
    self.er, self.er_min, self.er_decay = er, er_min, er_decay
    self.q = np.zeros((self.state_size, self.action_size), dtype="float32")

  def action(self, state: int, greedy: bool = True, q: np.ndarray | None = None) -> int:
    if q is None: q = self.q
    if not greedy and np.random.uniform() < self.er: return np.random.choice(self.action_size)
    else: return np.random.choice(np.flatnonzero(q[state] == q[state].max()))