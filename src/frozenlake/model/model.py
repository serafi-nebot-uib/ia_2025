import numpy as np
from gymnasium import Env
from frozenlake.const import DEBUG, STATS, SAMPLE

class Model:
  def __init__(self, state_size: int, action_size: int):
    self.name = "N/A"
    self.state_size, self.action_size = state_size, action_size
    self.rewards, self.time = [], []

  def action(self, state: int, greedy: bool = True) -> int: raise NotImplementedError()

  def train(self, env: Env, max_iter: int, threshold: float = 1e-9) -> int: raise NotImplementedError()

  def train_stats(self, iter: int, reward: float, t: float):
      if DEBUG > 0 or STATS > 0:
        self.rewards.append(reward)
        self.time.append(t)

      if iter % SAMPLE == 0 and DEBUG > 0:
        sr_avg = np.mean(self.rewards[-SAMPLE:])
        time_avg = np.mean(self.time[-SAMPLE:])
        time_total = np.sum(self.time[-SAMPLE:])
        print(f"{self.name:<10s} {iter:>7d} | sr: {sr_avg:>4.2f} | t: {time_total:.6f} s (total) {time_avg:.6f} s (avg)")

  def test(self, env: Env, num_iter: int = 1) -> float:
    s = 0.0
    for _ in range(num_iter):
      state, _ = env.reset()
      done, trunc, reward = False, False, 0
      while not (done or trunc): state, reward, done, trunc, _ = env.step(self.action(state))
      s += float(reward)
    return s / num_iter