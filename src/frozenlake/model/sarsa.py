from . import Model
import numpy as np
from gymnasium import Env

class SARSA(Model):
  def update(self, state: int, action: int, reward: float, new_state: int, new_action: int | None = None, final: bool = False):
    target = reward + self.dr * self.q[new_state, new_action] * (not final)
    self.q[state, action] += self.lr * (target - self.q[state, action])

  def train(self, env: Env, episodes: int, threshold: float = 1e-9) -> int:
    ep = 0
    while ep < episodes:
      state, _ = env.reset()
      action = self(state)
      reward = 0

      done, trunc = False, False
      while not (done or trunc):
        new_state, reward, done, trunc, _ = env.step(action)
        new_action = self(new_state, training=True)
        self.update(state, action, float(reward), new_state, new_action, done)
        state, action = new_state, new_action

      if reward != 0: self.er = max(self.er_min, self.er * self.er_decay)
      ep += 1
    return ep