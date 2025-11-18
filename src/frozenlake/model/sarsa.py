from . import Model
import numpy as np
from gymnasium import Env
from collections import deque

class SARSA(Model):
  def reset(self): self.q = np.zeros((self.state_size, self.action_size), dtype="float32")

  def __call__(self, state: int, *, training: bool = False) -> int:
    if training and np.random.uniform(0, 1) < self.er: return np.random.choice(self.action_size)
    else: return self.q[state].argmax()

  def update(self, state: int, action: int, reward: float, new_state: int, new_action: int | None = None, final: bool = False):
    target = reward + self.dr * self.q[new_state, new_action] * (not final)
    self.q[state, action] += self.lr * (target - self.q[state, action])

  def train(self, env: Env, episodes: int) -> int:
    ep = 0
    # while len(self.rewards_mean) <= 0 or self.rewards_mean[-1] <= 0.40:
    while ep < episodes: # or (len(self.rewards_mean) > 0 and self.rewards_mean[-1] <= 0.40):
      state, _ = env.reset()
      action = self(state)
      reward = 0
      done, trunc = False, False
      while not (done or trunc):
        new_state, reward, done, trunc, _ = env.step(action)
        new_action = self(new_state, training=True)
        self.update(state, action, reward, new_state, new_action, done)
        state, action = new_state, new_action
      if reward != 0: self.er = max(self.er_min, self.er * self.er_decay)
      ep += 1
    return ep