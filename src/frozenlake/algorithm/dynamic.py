import time
import numpy as np
from gymnasium import Env
from frozenlake.algorithm import Algorithm

class DynamicProgramming(Algorithm):
  def __init__(self, state_size: int, action_size: int, probs: dict[int, dict[int, list[tuple[float, int, float, bool]]]], dr: float):
    super().__init__(state_size, action_size)
    self.probs, self.dr = probs, dr
    self.q = np.zeros((self.state_size, self.action_size), dtype="float32")
    self.v = np.zeros(self.state_size, dtype="float32")

  def action(self, state: int, greedy: bool = True) -> int:
    """Choose which action to perform for the given `state` (`greedy` does nothing)"""
    return self.q[state].argmax()

  def train(self, env: Env, num_iter: int) -> int:
    iter = 0
    while iter < num_iter:
      iter_start = time.perf_counter()

      # for each state, iterate over its actions and update its value based on its transition probability
      for state in range(self.state_size):
        for action in range(self.action_size):
          # q(s, a) = \sum_{s',r}(p(s',r | s,a) \cdot (r + \gamma v(s')))
          self.q[state, action] = sum(prob * (reward + self.dr * self.v[new_state]) for (prob, new_state, reward, _) in self.probs[state][action])
        self.v[state] = np.max(self.q[state])

      iter += 1
      iter_end = time.perf_counter()
      # run 100 tests to get the average number of steps and sr for the current iteration
      steps, perf = map(np.mean, self.test(env, num_iter=100))
      self.train_stats(iter, steps.item(), perf.item(), iter_end - iter_start, sample=1)

    return iter