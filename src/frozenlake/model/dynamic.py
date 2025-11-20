import time
import numpy as np
from gymnasium import Env
from frozenlake.model import Model

class DynamicProgramming(Model):
  def __init__(self, state_size: int, action_size: int, probs: dict[int, dict[int, list[tuple[float, int, float, bool]]]], dr: float):
    super().__init__(state_size, action_size)
    self.probs, self.dr = probs, dr
    self.q = np.zeros((self.state_size, self.action_size), dtype="float32")
    self.v = np.zeros(self.state_size, dtype="float32")
    self.last_v = np.zeros(self.state_size, dtype="float32")

  def action(self, state: int, greedy: bool = True) -> int: return self.q[state].argmax()
  def update(self, state: int, action: int):
    self.q[state, action] = sum(prob * (reward + self.dr * self.last_v[new_state]) for (prob, new_state, reward, _) in self.probs[state][action])

  def train(self, env: Env, max_iter: int) -> int:
    iter = 0
    while iter < max_iter:
      iter_start = time.perf_counter()

      self.last_v = np.copy(self.v)
      for state in range(self.state_size):
        for action in range(self.action_size): self.update(state, action)
        self.v[state] = np.max(self.q[state])

      iter += 1
      iter_end = time.perf_counter()
      perf = self.test(env, num_iter=100)
      self.train_stats(iter, perf, iter_end - iter_start, sample=1)

    return iter