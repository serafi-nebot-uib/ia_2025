from frozenlake.model import Model
import numpy as np
from gymnasium import Env
from frozenlake.const import DEBUG, STATS
import time

class SARSA(Model):
  def reset(self):
    super().reset()
    self.q = np.zeros((self.state_size, self.action_size), dtype="float32")

  def update(self, state: int, action: int, reward: float, new_state: int | None = None, new_action: int | None = None, final: bool = False):
    target = reward + self.dr * self.q[new_state, new_action] * (not final)
    self.q[state, action] += self.lr * (target - self.q[state, action])
    self.policy = self.q.argmax(axis=-1)

  def train(self, env: Env, max_iter: int, threshold: float = 1e-9) -> int:
    iter = 0
    iter_start = time.perf_counter()
    while iter < max_iter:
      state, _ = env.reset()
      action = self(state)
      reward = 0

      done, trunc = False, False
      while not (done or trunc):
        new_state, reward, done, trunc, _ = env.step(action)
        new_action = self(new_state, greedy=False)
        self.update(state, action, float(reward), new_state, new_action, done)
        state, action = new_state, new_action

      if reward != 0: self.er = max(self.er_min, self.er * self.er_decay)
      iter += 1

      if STATS > 0: self.rewards.append(reward)
      if iter % 1000 == 0 and DEBUG > 0:
        iter_end = time.perf_counter()
        print(f"{iter:>7d} | {reward:>4.2f} : {self.policy} | {iter_end - iter_start:.6f} sec")
        iter_start = time.perf_counter()

    return iter