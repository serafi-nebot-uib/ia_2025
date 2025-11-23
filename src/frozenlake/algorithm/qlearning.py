import time
from gymnasium import Env
from frozenlake.algorithm import LearningAlgorithm

class QLearning(LearningAlgorithm):
  def update(self, state: int, action: int, reward: float, new_state: int, final: bool):
    target = reward + self.dr * self.q[new_state].max() * (not final)
    self.q[state, action] += self.lr * (target - self.q[state, action])

  def train(self, env: Env, num_iter: int) -> int:
    iter = 0
    while iter < num_iter:
      iter_start = time.perf_counter()
      state, _ = env.reset()
      reward, reward_total = 0.0, 0.0

      done, trunc, steps = False, False, 0
      while not (done or trunc):
        action = self.action(state, greedy=False) # epsilon-greedy action selection
        new_state, reward, done, trunc, _ = env.step(action)
        self.actions[action] += 1 # keep track of the chosen action
        reward = float(reward)
        self.update(state, action, reward, new_state, done) # update the Q table
        state = new_state
        reward_total += reward
        steps += 1

      self.er = max(self.er_min, self.er * self.er_decay)
      iter += 1
      iter_end = time.perf_counter()
      self.train_stats(iter, steps, reward_total, iter_end - iter_start)

    return iter