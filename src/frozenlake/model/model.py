import numpy as np
from gymnasium import Env
from typing import TypedDict, Unpack, Callable

# TODO: Parameters doesn't make sense for all models so it probably should be moved to a subclass
class Parameters(TypedDict):
  lr: float       # learning rate (alpha)
  dr: float       # discount rate (gamma)
  er: float       # exploration rate (epsilon)
  er_min: float   # exploration rate minumum
  er_decay: float # exploration rate decay

class Model:
  def __init__(self, state_size: int, action_size: int, **params: Unpack[Parameters]):
    self.state_size, self.action_size, self.params = state_size, action_size, params
    self.reset()

  def reset(self):
    self.lr, self.dr = self.params["lr"], self.params["dr"]
    self.er, self.er_min, self.er_decay = self.params["er"], self.params["er_min"], self.params["er_decay"]
    self.policy = np.zeros(self.state_size, dtype="uint8")

  def __call__(self, state: int, *, greedy: bool = True) -> int:
    return np.random.choice(self.action_size) if not greedy and np.random.uniform() < self.er else self.policy[state]

  def train(self, env: Env, max_iter: int, threshold: float = 1e-9) -> int: raise NotImplementedError()
  def test(self, env: Env, num_iter: int = 1, fn: Callable[[int], int] | None = None) -> float:
    if not callable(fn): fn = self
    s = 0.0
    for _ in range(num_iter):
      state, _ = env.reset()
      done, trunc, reward = False, False, 0
      while not (done or trunc): state, reward, done, trunc, _ = env.step(fn(state))
      s += float(reward)
    return s / num_iter