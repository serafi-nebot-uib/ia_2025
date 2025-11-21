import time
import numpy as np
from gymnasium import Env
from frozenlake.algorithm import LearningAlgorithm

class SARSA(LearningAlgorithm):
  def update(self, state: int, action: int, reward: float, new_state: int, new_action: int, final: bool):
    target = reward + self.dr * self.q[new_state, new_action] * (not final)
    self.q[state, action] += self.lr * (target - self.q[state, action])

  def train(self, env: Env, num_iter: int) -> int:
    iter = 0
    while iter < num_iter:
      iter_start = time.perf_counter()
      state, _ = env.reset()
      action = self.action(state, greedy=False)
      reward, reward_total = 0.0, 0.0

      done, trunc, steps = False, False, 0
      while not (done or trunc):
        new_state, reward, done, trunc, _ = env.step(action)
        self.actions[action] += 1
        reward = float(reward)
        new_action = self.action(new_state, greedy=False)
        self.update(state, action, float(reward), new_state, new_action, done)
        state, action = new_state, new_action
        reward_total += reward
        steps += 1

      self.er = max(self.er_min, self.er * self.er_decay)
      iter += 1
      iter_end = time.perf_counter()
      self.train_stats(iter, steps, reward_total, iter_end - iter_start)

    self.policy = self.q.argmax(axis=-1)

    return iter