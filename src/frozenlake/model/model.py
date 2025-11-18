import numpy as np
from gymnasium import Env
from typing import TypedDict, Unpack

class Parameters(TypedDict):
  lr: float       # learning rate (alpha)
  dr: float       # discount rate (gamma)
  er: float       # exploration rate (epsilon)
  er_min: float   # exploration rate minumum
  er_decay: float # exploration rate decay

class Model:
  def __init__(self, state_size: int, action_size: int, *,
               probs: dict[int, dict[int, list[tuple[float, int, float, bool]]]] | None = None,
               **params: Unpack[Parameters]):
    self.state_size, self.action_size, self.probs, self.params = state_size, action_size, probs, params
    self.reset()

  def reset(self):
    self.lr, self.dr = self.params["lr"], self.params["dr"]
    self.er, self.er_min, self.er_decay = self.params["er"], self.params["er_min"], self.params["er_decay"]
    self.q = np.zeros((self.state_size, self.action_size), dtype="float32")

  def __call__(self, state: int, *, training: bool = False) -> int:
    if training and np.random.uniform(0, 1) < self.er: return np.random.choice(self.action_size)
    else: return self.q[state].argmax().astype("uint8")

  def update(self, state: int, action: int, reward: float, new_state: int | None = None, new_action: int | None = None, final: bool = False): raise NotImplementedError()
  def train(self, env: Env, episodes: int, threshold: float = 1e-9) -> int: raise NotImplementedError()

  def printPolicy(self, env: Env):
      arrows = { 0: '⇐', 1: '⇓', 2: '⇒', 3: '⇑' }
      goal = "⊕"
      wall = " " # visible wall:"▓"
      v = self.q.max(axis=-1)
      policy = self.q.argmax(axis=-1)
      dim = int(np.sqrt(self.state_size))
      desc = env.unwrapped.desc
      print("       ╭────╮        ╭────╮   ")
      print("┍━━━━━━┥ v* ┝━━━━━┯━━┥ π* ┝━━┑")
      print("│      ╰────╯     │  ╰────╯  │")
      for y in range(dim):
          row_v = " ".join(f"{v[y*dim + x]:.1f}" for x in range(dim))
          row_pi = " ".join(wall if desc[y, x] == b'H' else goal if desc[y, x] == b'G' else arrows[policy[y*dim + x]] for x in range(dim))
          print(f"│ {row_v} │ {row_pi}  │")
      print("┕━━━━━━━━━━━━━━━━━┷━━━━━━━━━━┙")