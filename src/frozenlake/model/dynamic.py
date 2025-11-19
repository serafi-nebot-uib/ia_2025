from .model import Model, Parameters
import numpy as np
from gymnasium import Env
from typing import Unpack

class DynamicProgramming(Model):
  def __init__(self, state_size: int, action_size: int, probs: dict[int, dict[int, list[tuple[float, int, float, bool]]]], **params: Unpack[Parameters]):
    super().__init__(state_size, action_size, **params)
    self.probs = probs

  def reset(self): 
    super().reset()
    self.q = np.zeros((self.state_size, self.action_size), dtype="float32")
    self.v = np.zeros(self.state_size, dtype="float32")
    self.last_v = np.zeros(self.state_size, dtype="float32")

  def update(self, state: int, action: int, reward: float, new_state: int | None = None, new_action: int | None = None, final: bool = False):
    self.q[state, action] = sum(prob * (reward + self.dr * self.last_v[new_state]) for (prob, new_state, reward, _) in self.probs[state][action])

  def train(self, env: Env, max_iter: int, threshold: float = 1e-9) -> int:
    """ Value Iteration algorithm. Iterate until convergence falls below the threshold or the maximum number of iterations is reached. """
    diff = threshold + 1e-09
    ep = 0
    while diff > threshold and ep < max_iter:
        diff = 0
        self.last_v = np.copy(self.v)
        for state in range(self.state_size):
            for action in range(self.action_size): self.update(state, action, 0)
            self.v[state] = np.max(self.q[state])
            diff = max(diff, abs(self.last_v[state] - self.v[state]))
        ep += 1
        # print(f"diff: {diff} (threshold: {threshold})")
    print(f"\n{ep} iterations needed.\n" if ep < max_iter else f"\nIteration limit exceeded({max_iter}).\n")
    return ep