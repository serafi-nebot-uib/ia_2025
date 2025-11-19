import time
import numpy as np
from gymnasium import Env
from frozenlake.model import Model
from frozenlake.const import DEBUG, STATS, SAMPLE

class MonteCarlo(Model):
  def __init__(self, state_size: int, action_size: int, dr: float, er: float, er_min: float, er_decay: float):
    super().__init__(state_size, action_size)
    self.dr = dr
    self.er, self.er_min, self.er_decay = er, er_min, er_decay
    self.q = np.zeros((self.state_size, self.action_size), dtype="float32")
    self.c = np.zeros((self.state_size, self.action_size), dtype=int)

  def action(self, state: int, greedy: bool = True) -> int:
    return np.random.choice(self.action_size) if not greedy and np.random.uniform() < self.er else self.q[state].argmax()

  def update(self, state: int, action: int, reward: float):
    # new_avg = old_avg + (new_value - old_avg) / (n + 1)
    self.c[state][action] += 1
    self.q[state][action] += (reward - self.q[state][action]) / self.c[state][action]

  def _gen_episode(self, env: Env, state: int):
    episode = []
    done, trunc = False, False
    while not (done or trunc):
      action = self.action(state, greedy=False)
      new_state, reward, done, trunc, _ = env.step(action)
      episode.append((state, action, reward))
      state = new_state
    return episode

  def train(self, env: Env, max_iter: int, threshold: float = 1e-9) -> int:
    iter = 0
    iter_start = time.perf_counter()
    while iter < max_iter:
      state, _ = env.reset()
      g = 0
      visited = set()
      episode = self._gen_episode(env, state) 

      reward_total = 0.0
      for state, action, reward in episode[::-1]:
        reward_total += reward
        g = reward + self.dr * g
        sa = (state, action)
        if sa not in visited:
          visited.add(sa)
          self.update(state, action, g)
      if reward_total > 0: self.er = max(self.er_min, self.er * self.er_decay)
      self.policy = self.q.argmax(axis=-1)

      iter += 1

      if STATS > 0: self.rewards.append(reward_total)
      if iter % SAMPLE == 0 and DEBUG > 0:
        iter_end = time.perf_counter()
        avg = np.mean(self.rewards[-SAMPLE:])
        print(f"{iter:>7d} | {avg:>4.2f} | {iter_end - iter_start:.6f} sec")
        iter_start = time.perf_counter()

    return max_iter