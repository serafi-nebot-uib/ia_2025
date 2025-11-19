from frozenlake.model import Model
import numpy as np
from gymnasium import Env
from frozenlake.const import DEBUG, STATS, SAMPLE
import time

class SARSA(Model):
  def __call__(self, state: int, *, greedy: bool = True, training: bool = False) -> int:
    action = self.q[state].argmax() if training else self.policy[state]
    return np.random.choice(self.action_size) if not greedy and np.random.uniform() < self.er else action

  def reset(self):
    super().reset()
    self.q = np.zeros((self.state_size, self.action_size), dtype="float32")

  def update(self, state: int, action: int, reward: float, new_state: int | None = None, new_action: int | None = None, final: bool = False):
    target = reward + self.dr * self.q[new_state, new_action] * (not final)
    self.q[state, action] += self.lr * (target - self.q[state, action])

  def train(self, env: Env, max_iter: int, threshold: float = 1e-9) -> int:
    iter = 0
    iter_start = time.perf_counter()
    while iter < max_iter:
      state, _ = env.reset()
      action = self(state)
      reward, reward_total = 0.0, 0.0

      done, trunc = False, False
      while not (done or trunc):
        new_state, reward, done, trunc, _ = env.step(action)
        reward = float(reward)
        new_action = self(new_state, greedy=False, training=True)
        self.update(state, action, float(reward), new_state, new_action, done)
        state, action = new_state, new_action
        reward_total += reward

      if reward_total > 0: self.er = max(self.er_min, self.er * self.er_decay)
      iter += 1

      if DEBUG > 0 or STATS > 0: self.rewards.append(reward_total)
      if iter % SAMPLE == 0 and DEBUG > 0:
        iter_end = time.perf_counter()
        avg = np.mean(self.rewards[-SAMPLE:])
        print(f"{iter:>7d} | {avg:>4.2f} | {iter_end - iter_start:.6f} sec")
        iter_start = time.perf_counter()

    self.policy = self.q.argmax(axis=-1)

    return iter