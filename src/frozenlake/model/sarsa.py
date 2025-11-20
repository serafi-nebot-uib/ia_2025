import time
import numpy as np
from gymnasium import Env
from frozenlake.model import Model

class SARSA(Model):
  def __init__(self, state_size: int, action_size: int, lr: float, dr: float, er: float, er_min: float, er_decay: float):
    super().__init__(state_size, action_size)
    self.name = "SARSA"
    self.lr, self.dr = lr, dr
    self.er, self.er_min, self.er_decay = er, er_min, er_decay
    self.q = np.zeros((self.state_size, self.action_size), dtype="float32")

  def action(self, state: int, greedy: bool = True) -> int:
    return np.random.choice(self.action_size) if not greedy and np.random.uniform() < self.er else self.q[state].argmax()

  def update(self, state: int, action: int, reward: float, new_state: int, new_action: int, final: bool):
    target = reward + self.dr * self.q[new_state, new_action] * (not final)
    self.q[state, action] += self.lr * (target - self.q[state, action])

  def train(self, env: Env, max_iter: int, threshold: float = 1e-9) -> int:
    iter = 0
    while iter < max_iter:
      iter_start = time.perf_counter()
      state, _ = env.reset()
      action = self.action(state, greedy=False)
      reward, reward_total = 0.0, 0.0

      done, trunc = False, False
      while not (done or trunc):
        new_state, reward, done, trunc, _ = env.step(action)
        reward = float(reward)
        new_action = self.action(new_state, greedy=False)
        self.update(state, action, float(reward), new_state, new_action, done)
        state, action = new_state, new_action
        reward_total += reward

      self.er = max(self.er_min, self.er * self.er_decay)
      iter += 1
      iter_end = time.perf_counter()
      self.train_stats(iter, reward_total, iter_end - iter_start)

    self.policy = self.q.argmax(axis=-1)

    return iter