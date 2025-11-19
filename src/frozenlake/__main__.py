from frozenlake.model import Parameters, Model, SARSA, QLearning, DoubleQLearning, MonteCarlo, DynamicProgramming, Genetic
import frozenlake.plot as plot
import numpy as np
import gymnasium as gym
from gymnasium import Env
from tqdm import tqdm
import matplotlib.pyplot as plt
import random

SLIPPERY = True

def printPolicy(env: Env, policy: np.ndarray):
    arrows, goal, wall = [ '⇐', '⇓', '⇒', '⇑' ], "⊕", " " # visible wall:"▓"
    dim = int(np.sqrt(env.observation_space.n))
    desc = env.unwrapped.desc
    print("   ╭────╮   ")
    print("┍━━┥ π* ┝━━┑")
    print("│  ╰────╯  │")
    for y in range(dim):
        row_pi = " ".join(wall if desc[y, x] == b'H' else goal if desc[y, x] == b'G' else arrows[policy[y*dim + x]] for x in range(dim))
        print(f"│ {row_pi}  │")
    print("┕━━━━━━━━━━┙")

# def printValue(self, env: Env):
#     arrows = { 0: '⇐', 1: '⇓', 2: '⇒', 3: '⇑' }
#     goal = "⊕"
#     wall = " " # visible wall:"▓"
#     v = self.q.max(axis=-1)
#     policy = self.q.argmax(axis=-1)
#     dim = int(np.sqrt(self.state_size))
#     desc = env.unwrapped.desc
#     print("       ╭────╮        ╭────╮   ")
#     print("┍━━━━━━┥ v* ┝━━━━━┯━━┥ π* ┝━━┑")
#     print("│      ╰────╯     │  ╰────╯  │")
#     for y in range(dim):
#         row_v = " ".join(f"{v[y*dim + x]:.1f}" for x in range(dim))
#         row_pi = " ".join(wall if desc[y, x] == b'H' else goal if desc[y, x] == b'G' else arrows[policy[y*dim + x]] for x in range(dim))
#         print(f"│ {row_v} │ {row_pi}  │")
#     print("┕━━━━━━━━━━━━━━━━━┷━━━━━━━━━━┙")

def test(model: Model, *, num_iter, seed: int | None = None):
  env = gym.make("FrozenLake-v1", is_slippery=SLIPPERY, render_mode=None)

  if seed:
    np.random.seed(seed)
    random.seed(seed)

  env.reset(seed=seed)
  num_iter = model.train(env, num_iter)
  sr = model.test(env, 1000)
  env.close()

  return sr, num_iter

if __name__ == "__main__":
  # TODO: is there a better way to retrieve/build P without making a new environment just for it?
  env = gym.make("FrozenLake-v1", is_slippery=SLIPPERY, render_mode=None)
  probs = env.unwrapped.P
  state_size = env.observation_space.n
  action_size = env.action_space.n
  env.close()

  # m = DynamicProgramming(state_size=4*4, action_size=4, probs=probs, **params)
  # m = Genetic(state_size=4*4, action_size=4, population_size=100, mutation_rate=0.08, **params)
  params = Parameters(lr=0.20,
                      dr=0.95,
                      er=1.00, er_min=0.01, er_decay=0.95)
  models = [
         MonteCarlo(state_size, action_size, lr=0.10, dr=1.00, er=1.00, er_min=0.05, er_decay=0.99995),
    #           SARSA(state_size, action_size, lr=0.20, dr=0.95, er=1.00, er_min=0.01, er_decay=0.95),
    #       QLearning(state_size, action_size, lr=0.20, dr=0.95, er=1.00, er_min=0.01, er_decay=0.95),
    # DoubleQLearning(state_size, action_size, lr=0.20, dr=0.95, er=1.00, er_min=0.01, er_decay=0.95)
  ]
  data = {}
  for model in models:
    sr, ep = test(model, num_iter=20000)
    print(f"sr: {sr:.4f}; ep: {ep}")
    data[model.__class__.__name__] = model.rewards
  fig = plot.success_rate(data, 1000)
  plt.show()

  # desc = env.unwrapped.desc
  # plot.policy(m.policy, m.q.max(axis=-1))
  # fig = plot.qtable(desc, m.q)
  # fig.suptitle("QLearning")
  # fig.tight_layout()
  # plt.show()

  # sr, ep = [], []
  # for i in tqdm(range(25), desc="test"):
  #   s, e = test(QLearning(**params))
  #   sr.append(s)
  #   ep.append(e)
  # print(f"avg success rate: {np.mean(sr):.4f}")
  # fig, ax1 = plt.subplots(figsize=(8, 8))
  # ax1.plot(np.arange(len(sr)), sr, label="sr", color="blue")
  # # ax2 = ax1.twinx()
  # # ax2.plot(np.arange(len(ep)), ep, label="ep", color="red")
  # plt.legend(loc="upper right")
  # plt.show()

  # view Q table (policy)
  # requirements:
  #     - uv pip install matplotlib
  # ----- for SARSA/QLearning/MonteCarlo
  # plot.qtable(m.q, 4, 4)
  # plot.policy(m.q, 4, 4)
  # ----- for DoubleQLearning
  # plot.qtable(m.qa, 4, 4)
  # plot.qtable(m.qb, 4, 4)

  # env = gym.make("FrozenLake-v1", is_slippery=SLIPPERY, render_mode="human")
  # state, _ = env.reset()
  # term, trunc = False, False
  # while not (term or trunc): state, _ , term, trunc, _ = env.step(m(state))
  # env.close()