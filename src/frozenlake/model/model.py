from gymnasium import Env

class Model:
  def __init__(self, state_size: int, action_size: int):
    self.state_size, self.action_size = state_size, action_size
    self.rewards = []

  def action(self, state: int, greedy: bool = True) -> int: raise NotImplementedError()
  def train(self, env: Env, max_iter: int, threshold: float = 1e-9) -> int: raise NotImplementedError()
  def test(self, env: Env, num_iter: int = 1) -> float:
    s = 0.0
    for _ in range(num_iter):
      state, _ = env.reset()
      done, trunc, reward = False, False, 0
      while not (done or trunc): state, reward, done, trunc, _ = env.step(self.action(state))
      s += float(reward)
    return s / num_iter