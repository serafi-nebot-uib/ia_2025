from .model import Model, Parameters
import numpy as np
from gymnasium import Env
from typing import TypedDict, Unpack

class DynamicProgramming(Model):
  def reset(self): 
    super().reset()
    self.v = np.zeros(self.state_size, dtype="float32")
    self.last_v = np.copy(self.v)
    self.p = None

  def update(self, state: int, action: int):
    if self.probs is None: return
    self.q[state, action] = sum(prob * (reward + self.dr * self.last_v[new_state]) for (prob, new_state, reward, _) in self.probs)

  def train(self, env: Env, episodes: int, threshold: float = 1e-9) -> int:
    """ Value Iteration algorithm. Iterate until convergence falls below the threshold or the maximum number of iterations is reached. """
    diff = threshold + 1e-09
    ep = 0
    while diff > threshold and ep < episodes:
        diff = 0
        self.last_v = np.copy(self.v)
        for state in range(self.state_size):
            for action in range(self.action_size): self.update(state, action)
            self.v[state] = np.max(self.q[state])
            diff = max(diff, abs(self.last_v[state] - self.v[state]))
        ep += 1
        # print(f"diff: {diff} (threshold: {threshold})")
    print(f"\n{ep} iterations needed.\n" if ep < episodes else f"\nIteration limit exceeded({episodes}).\n")
    return ep