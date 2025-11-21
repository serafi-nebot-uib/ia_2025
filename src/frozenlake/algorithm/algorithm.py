import numpy as np
from gymnasium import Env
from frozenlake.const import DEBUG, STATS, SAMPLE

class Algorithm:
  def __init__(self, state_size: int, action_size: int):
    self.name = self.__class__.__name__
    self.state_size, self.action_size = state_size, action_size
    self.time, self.performance = [], []

  def action(self, state: int, greedy: bool = True) -> int: raise NotImplementedError()
  def train(self, env: Env, num_iter: int) -> int: raise NotImplementedError()

  def train_stats(self, iter: int, reward: float, t: float, sample: int = SAMPLE):
    if DEBUG > 0 or STATS > 0:
      self.performance.append(reward)
      self.time.append(t)

    if DEBUG > 0 and (not sample or iter % sample == 0):
      perf_avg = np.mean(self.performance[-sample:])
      time_avg = np.mean(self.time[-sample:])
      time_total = np.sum(self.time[-sample:])
      print(f"{self.name:<15} {iter:>7d} | perf: {perf_avg:>4.2f} | t: {time_total:.6f} s", end="")
      if sample > 1: print(f" (total) {time_avg:.6f} s (avg)", end="")
      print(flush=True)

  def run(self, env: Env) -> tuple[int, float]:
    state, _ = env.reset()
    steps, reward_total = 0, 0
    done, trunc = False, False
    while not (done or trunc):
      state, reward, done, trunc, _ = env.step(self.action(state, greedy=True))
      reward_total += float(reward)
      steps += 1
    return steps, reward_total

  def test(self, env: Env, num_iter: int) -> tuple[list[int], list[int]]:
    steps, success = [], []
    for _ in range(num_iter):
      step, reward = self.run(env)
      steps.append(step)
      success.append(int(reward > 0))
    return steps, success

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