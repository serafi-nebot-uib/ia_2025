import time
import numpy as np
from gymnasium import Env
from frozenlake.model import Model

class MonteCarlo(Model):
  def __init__(self, state_size: int, action_size: int, dr: float, er: float, er_min: float, er_decay: float):
    super().__init__(state_size, action_size)
    self.name = "MC"
    self.dr = dr
    self.er, self.er_min, self.er_decay = er, er_min, er_decay
    self.q = np.zeros((self.state_size, self.action_size), dtype="float32")
    self.c = np.zeros((self.state_size, self.action_size), dtype=int)

  def action(self, state: int, greedy: bool = True) -> int:
    if not greedy and np.random.uniform() < self.er: return np.random.choice(self.action_size)
    else: return np.random.choice(np.flatnonzero(self.q[state] == self.q[state].max()))

  def update(self, state: int, action: int, g: float):
    # new_avg = old_avg + (new_value - old_avg) / (n + 1)
    self.c[state][action] += 1
    self.q[state][action] += (g - self.q[state][action]) / self.c[state][action]

  def _gen_episode(self, env: Env):
    state, _ = env.reset()
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
    while iter < max_iter:
      iter_start = time.perf_counter()
      episode = self._gen_episode(env) 

      reward_total = 0.0
      g = 0.0
      visited = set()
      for state, action, reward in reversed(episode):
        reward_total += reward
        g = reward + self.dr * g
        sa = (state, action)
        if sa not in visited:
          visited.add(sa)
          self.update(state, action, g)

      self.er = max(self.er_min, self.er * self.er_decay)
      iter += 1
      iter_end = time.perf_counter()
      self.train_stats(iter, reward_total, iter_end - iter_start)

    return max_iter