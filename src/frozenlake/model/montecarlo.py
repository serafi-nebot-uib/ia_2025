from frozenlake.model import Model
import numpy as np
from gymnasium import Env

class MonteCarlo(Model):
  def reset(self):
    super().reset()
    self.q = np.zeros((self.state_size, self.action_size), dtype="float32")
    self.c = np.zeros((self.state_size, self.action_size), dtype=int)

  def update(self, state: int, action: int, reward: float, new_state: int | None = None, new_action: int | None = None, final: bool = False):
    # new_avg = old_avg + (new_value - old_avg) / (n + 1)
    self.c[state][action] += 1
    self.q[state][action] += (reward - self.q[state][action]) / self.c[state][action]
  
  def _gen_episode(self, env: Env, state: int):
    episode = []
    done, trunc = False, False
    while not (done or trunc):
      action = self(state, greedy=False)
      new_state, reward, done, trunc, _ = env.step(action)
      episode.append((state, action, reward))
      state = new_state
    return episode

  def train(self, env: Env, max_iter: int, threshold: float = 1e-9) -> int:
    for _ in range(max_iter):
      state, _ = env.reset()
      g = 0
      visited = set()
      episode = self._gen_episode(env, state) 
      for state, action, reward in episode[::-1]:
        g = reward + self.dr * g
        sa = (state, action)
        if sa not in visited:
          visited.add(sa)
          self.update(state, action, g)
      self.er = max(self.er_min, self.er * self.er_decay)
    return max_iter