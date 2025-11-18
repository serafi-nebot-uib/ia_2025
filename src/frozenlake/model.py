import numpy as np
from gymnasium import Env
from typing import Unpack, TypedDict
import random
from collections import deque

class Parameters(TypedDict):
  lr: float # learning rate (alpha)
  dr: float # discount rate (gamma)
  er: float # exploration rate (epsilon)
  er_min: float
  er_decay: float

class Model:
  def __init__(self, state_size: int, action_size: int, **params: Unpack[Parameters]):
    self.state_size, self.action_size = state_size, action_size
    self.lr, self.dr = params["lr"], params["dr"]
    self.er, self.er_min, self.er_decay = params["er"], params["er_min"], params["er_decay"]
    self.reset()

  def __call__(self, state: int, *, training: bool = False) -> int: raise NotImplementedError()
  def reset(self): pass

class QLearning(Model):
  def reset(self): self.q = np.zeros((self.state_size, self.action_size), dtype="float32")

  def __call__(self, state: int, *, training: bool = False) -> int:
    if training and np.random.uniform(0, 1) < self.er: return np.random.choice(self.action_size)
    else: return self.q[state].argmax()

  def train(self, env: Env, episodes: int, *, seed: int | None = None):
    ep = 0
    self.rewards = deque(maxlen=1000)
    self.rewards_mean = []
    while ep < episodes: # or (len(self.rewards_mean) > 0 and self.rewards_mean[-1] <= 0.40):
      state, _ = env.reset()
      reward = 0

      done, trunc = False, False
      while not (done or trunc):
        action = self(state, training=True)
        new_state, reward, done, trunc, _ = env.step(action)

        # update Q table
        target = float(reward) + self.dr * self.q[new_state].max() * (not done)
        self.q[state, action] += self.lr * (target - self.q[state, action])

        state = new_state

      if reward != 0: self.er = max(self.er_min, self.er * self.er_decay)
      ep += 1
      self.rewards.append(reward)
      self.rewards_mean.append(np.mean(self.rewards).item())
    return ep

class SARSA(Model):
  def reset(self): self.q = np.zeros((self.state_size, self.action_size), dtype="float32")

  def __call__(self, state: int, *, training: bool = False) -> int:
    if training and np.random.uniform(0, 1) < self.er: return np.random.choice(self.action_size)
    else: return self.q[state].argmax()

  def update(self, state: int, action: int, reward: float, new_state: int, new_action: int, final: bool = False):
    target = reward + self.dr * self.q[new_state, new_action] * (not final)
    self.q[state, action] += self.lr * (target - self.q[state, action])

  def train(self, env: Env, episodes: int):
    ep = 0
    self.rewards = deque(maxlen=1000)
    self.rewards_mean = []
    # while len(self.rewards_mean) <= 0 or self.rewards_mean[-1] <= 0.40:
    while ep < episodes: # or (len(self.rewards_mean) > 0 and self.rewards_mean[-1] <= 0.40):
      state, _ = env.reset()
      action = self(state)
      reward = 0
      term, trunc = False, False
      while not (term or trunc):
        new_state, reward, term, trunc, _ = env.step(action)
        new_action = self(new_state, training=True)
        self.update(state, action, reward, new_state, new_action, term)
        state, action = new_state, new_action
      if reward != 0: self.er = max(self.er_min, self.er * self.er_decay)
      ep += 1
      self.rewards.append(reward)
      self.rewards_mean.append(np.mean(self.rewards).item())
    return ep

# class DoubleQLearning(Model): pass
#   def reset(self):
#     self.qa = np.zeros((self.state_size, self.action_size), dtype="float32")
#     self.qb = np.zeros((self.state_size, self.action_size), dtype="float32")

#   def __call__(self, state: int, *, training: bool = False) -> int:
#     if training and np.random.uniform(0, 1) < self.er: return np.random.choice(self.action_size)
#     else: return np.argmax((self.qa[state] + self.qb[state]) / 2).astype("int8")

#   def update(self, state: int, action: int, reward: float, new_state: int, final: bool = False):
#     if np.random.uniform(0, 1) < 0.5:
#       target = reward + self.dr * self.qb[new_state].max() * (not final)
#       td_error = target - self.qa[state, action]
#       self.qa[state, action] += self.lr * td_error
#     else:
#       target = reward + self.dr * self.qa[new_state].max() * (not final)
#       td_error = target - self.qb[state, action]
#       self.qb[state, action] += self.lr * td_error

#   def train(self, env: Env, episodes: int):
#     for _ in range(episodes):
#       state, _ = env.reset()
#       term, trunc = False, False
#       reward = 0
#       while not (term or trunc):
#         action = self(state, training=True)
#         new_state, reward, term, trunc, _ = env.step(action)
#         self.update(state, action, reward, new_state, term)
#         state = new_state
#       if reward != 0: self.er = max(self.er_min, self.er * self.er_decay)