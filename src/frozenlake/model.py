import numpy as np
from gymnasium import Env
from typing import Unpack, TypedDict, Callable
import random
from collections import deque

class Parameter:
  def __init__(self, value: float, min: float | None = None, decay: float | Callable[[float], float] | None = None):
    self.value, self.min = value, min
    self.decay = (lambda x: decay * x) if not callable(decay) else decay

  def step(self):
    if self.decay:
      value = self.decay(self.value)
      self.value = min(self.value, value) if self.min else value

  def __enter__(self) -> float: return self.value
  def __exit__(self, *args, **kwargs): self.step()
  def __call__(self, x: float) -> float:
    with self as v: return v * x

class Parameters(TypedDict):
  lr: float             # learning rate (alpha)
  dr: float             # discount rate (gamma)
  er: float             # exploration rate (epsilon)
  er_min: float
  er_decay: float

# TODO: is there a better way to assign the parameters to instance variables without a base class?
class Model:
  def __init__(self, state_size: int, action_size: int, **kwargs: Unpack[Parameters]):
    self.state_size, self.action_size = state_size, action_size
    self.lr, self.dr = kwargs["lr"], kwargs["dr"]
    self.er, self.er_min, self.er_decay = kwargs["er"], kwargs["er_min"], kwargs["er_decay"]
    self.reset()

  def __call__(self, state: int, *, training: bool = False) -> int: raise NotImplementedError()
  def reset(self): pass

class SARSA(Model):
  def reset(self): self.q = np.zeros((self.state_size, self.action_size), dtype="float32")

  def __call__(self, state: int, *, training: bool = False) -> int:
    if training and np.random.uniform(0, 1) < self.er: return np.random.choice(self.action_size)
    else: return self.q[state].argmax()

  def update(self, state: int, action: int, reward: float, new_state: int, new_action: int, final: bool = False):
    target = reward + self.dr * self.q[new_state, new_action] * (not final)
    self.q[state, action] += self.lr * (target - self.q[state, action])

  def train(self, env: Env, episodes: int):
    for _ in range(episodes):
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

class QLearning(Model):
  def reset(self): self.q = np.zeros((self.state_size, self.action_size), dtype="float32")

  def __call__(self, state: int, *, training: bool = False) -> int:
    if training and np.random.uniform(0, 1) < self.er: return np.random.choice(self.action_size)
    else: return self.q[state].argmax()

  def train(self, env: Env, episodes: int, *, seed: int | None = None):
    if seed is not None:
      random.seed(seed)
      np.random.seed(seed)
      env.action_space.seed(seed)
      env.reset(seed=seed)

    ep = 0
    self.rewards = deque(maxlen=1000)
    self.rewards_mean = []
    while ep < episodes: # or (len(self.rewards_mean) > 0 and self.rewards_mean[-1] <= 0.40):
      state, _ = env.reset() if seed is None else env.reset(seed=seed + ep)
      reward = 0

      done, trunc = False, False
      while not (done or trunc):
        action = self(state, training=True)
        new_state, reward, done, trunc, _ = env.step(action)

        # update Q table
        target = reward + self.dr * self.q[new_state].max() * (not done)
        self.q[state, action] += self.lr * (target - self.q[state, action])

        state = new_state

      if reward != 0: self.er = max(self.er_min, self.er * self.er_decay)
      ep += 1
      self.rewards.append(reward)
      self.rewards_mean.append(np.mean(self.rewards).item())
    return ep

class DoubleQLearning(Model):
  def reset(self):
    self.qa = np.zeros((self.state_size, self.action_size), dtype="float32")
    self.qb = np.zeros((self.state_size, self.action_size), dtype="float32")

  def __call__(self, state: int, *, training: bool = False) -> int:
    if training and np.random.uniform(0, 1) < self.er: return np.random.choice(self.action_size)
    else: return np.argmax((self.qa[state] + self.qb[state]) / 2).astype("int8")

  def update(self, state: int, action: int, reward: float, new_state: int, final: bool = False):
    if np.random.uniform(0, 1) < 0.5:
      target = reward + self.dr * self.qb[new_state].max() * (not final)
      td_error = target - self.qa[state, action]
      self.qa[state, action] += self.lr * td_error
    else:
      target = reward + self.dr * self.qa[new_state].max() * (not final)
      td_error = target - self.qb[state, action]
      self.qb[state, action] += self.lr * td_error

  def train(self, env: Env, episodes: int):
    for _ in range(episodes):
      state, _ = env.reset()
      term, trunc = False, False
      reward = 0
      while not (term or trunc):
        action = self(state, training=True)
        new_state, reward, term, trunc, _ = env.step(action)
        self.update(state, action, reward, new_state, term)
        state = new_state
      if reward != 0: self.er = max(self.er_min, self.er * self.er_decay)