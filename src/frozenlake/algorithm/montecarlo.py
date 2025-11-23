import time
import numpy as np
from gymnasium import Env
from frozenlake.algorithm import LearningAlgorithm

class MonteCarlo(LearningAlgorithm):
  def __init__(self, state_size: int, action_size: int, dr: float, er: float, er_min: float, er_decay: float):
    super().__init__(state_size, action_size, 0.00, dr, er, er_min, er_decay)
    # there's no need to keep all of the returns to calculate the average,
    # can be replaced with incremental averaging if you track the state-action visit count
    # C table to keep track the amount of times every action has been chosen for every state
    self.c = np.zeros((self.state_size, self.action_size), dtype=int)

  def update(self, state: int, action: int, g: float):
    # new_avg = old_avg + (new_value - old_avg) / (n + 1)
    self.c[state, action] += 1
    self.q[state, action] += (g - self.q[state, action]) / self.c[state, action]

  def _gen_episode(self, env: Env):
    """Generate a complete episode and return all of the `(state, action, reward)` sequences"""
    state, _ = env.reset()
    episode = []
    done, trunc = False, False
    while not (done or trunc):
      action = self.action(state, greedy=False) # epsilon-greedy action selection
      new_state, reward, done, trunc, _ = env.step(action)
      episode.append((state, action, reward)) # save the current transition
      state = new_state
    return episode

  def train(self, env: Env, num_iter: int) -> int:
    iter = 0
    while iter < num_iter:
      iter_start = time.perf_counter()
      episode = self._gen_episode(env) 

      # for the generated episode, walk back and calculate the discounted accumulated reward
      reward_total, g = 0.0, 0.0
      visited = set() # first-visit implementation, keep track of which (state, action) have been seen before
      for state, action, reward in reversed(episode):
        self.actions[action] += 1
        reward_total += reward
        g = reward + self.dr * g # discount last accumulated reward and add reward
        sa = (state, action)
        if sa not in visited:
          visited.add(sa)
          self.update(state, action, g)

      self.er = max(self.er_min, self.er * self.er_decay)
      iter += 1
      iter_end = time.perf_counter()
      self.train_stats(iter, len(episode), reward_total, iter_end - iter_start)

    return num_iter