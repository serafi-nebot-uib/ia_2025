import random
import numpy as np
import gymnasium as gym
from gymnasium import Env
import matplotlib.pyplot as plt
import frozenlake.plot as plot
from frozenlake.algorithm import Algorithm, SARSA, QLearning, DoubleQLearning, MonteCarlo, DynamicProgramming, Genetic

SLIPPERY = True

env = gym.make("FrozenLake-v1", is_slippery=SLIPPERY, render_mode=None)
DESC = env.unwrapped.desc
PROBS = env.unwrapped.P
STATE_SIZE = env.observation_space.n
ACTION_SIZE = env.action_space.n
env.close()

LEARNING = [
  (MonteCarlo,      { "state_size": STATE_SIZE, "action_size": ACTION_SIZE,             "dr": 0.99, "er": 1.00, "er_min": 0.01, "er_decay": 0.9995 }),
  (SARSA,           { "state_size": STATE_SIZE, "action_size": ACTION_SIZE, "lr": 0.20, "dr": 0.99, "er": 1.00, "er_min": 0.01, "er_decay": 0.9995 }),
  (QLearning,       { "state_size": STATE_SIZE, "action_size": ACTION_SIZE, "lr": 0.20, "dr": 0.99, "er": 1.00, "er_min": 0.01, "er_decay": 0.9995 }),
  (DoubleQLearning, { "state_size": STATE_SIZE, "action_size": ACTION_SIZE, "lr": 0.20, "dr": 0.99, "er": 1.00, "er_min": 0.01, "er_decay": 0.9995 }),
]

SEARCH = [
  (Genetic, { "state_size": STATE_SIZE, "action_size": ACTION_SIZE, "population_size": 100, "mutation_rate": 0.08 }),
  (DynamicProgramming, { "state_size": STATE_SIZE, "action_size": ACTION_SIZE, "probs": PROBS, "dr": 0.95 }),
]

# def printPolicy(env: Env, policy: np.ndarray):
#     arrows, goal, wall = [ '⇐', '⇓', '⇒', '⇑' ], "⊕", " " # visible wall:"▓"
#     dim = int(np.sqrt(state_size)
#     print("   ╭────╮   ")
#     print("┍━━┥ π* ┝━━┑")
#     print("│  ╰────╯  │")
#     for y in range(dim):
#         row_pi = " ".join(wall if desc[y, x] == b'H' else goal if desc[y, x] == b'G' else arrows[policy[y*dim + x]] for x in range(dim))
#         print(f"│ {row_pi}  │")
#     print("┕━━━━━━━━━━┙")

def train_test(alg: Algorithm, *, num_iter, seed: int | None = None):
  env = gym.make("FrozenLake-v1", is_slippery=SLIPPERY, render_mode=None)

  if seed:
    np.random.seed(seed)
    random.seed(seed)

  env.reset(seed=seed)
  num_iter = alg.train(env, num_iter)
  sr = alg.test(env, 1000)
  env.close()

  return sr, num_iter

def train_learning():
  algs = [c(**p) for c, p in LEARNING]
  data_sr, data_t = {}, {}
  for alg in algs:
    sr, ep = train_test(alg, num_iter=5000)
    print(f"sr: {sr:.4f}; ep: {ep}")
    data_sr[alg.name] = alg.performance
    data_t[alg.name] = alg.time

  plot.success_rate(data_sr, 1000)
  plot.train_time(data_t, 1000)
  for alg in algs: plot.qtable(alg.name, alg.q, DESC)
  plt.show()

def train_search():
  algs = [c(**p) for c, p in SEARCH]
  data_sr, data_t = {}, {}
  for alg in algs:
    sr, ep = train_test(alg, num_iter=40)
    print(f"sr: {sr:.4f}; ep: {ep}")
    data_sr[alg.name] = alg.performance
    data_t[alg.name] = alg.time

  plot.success_rate(data_sr)
  plot.train_time(data_t)
  plt.show()

def steps_learning():
  env = gym.make("FrozenLake-v1", is_slippery=SLIPPERY, render_mode=None)
  algs = [c(**p) for c, p in LEARNING]
  categories, values = [], []
  for alg in algs:
    steps = sum(alg.run(env)[0] for _ in range(1000)) / 1000
    categories.append(alg.name)
    values.append(steps)
  env.close()
  plt.bar(categories, values)
  plt.show()

if __name__ == "__main__":
  steps_learning()

  # env = gym.make("FrozenLake-v1", is_slippery=SLIPPERY, render_mode="human")
  # state, _ = env.reset()
  # term, trunc = False, False
  # while not (term or trunc): state, _ , term, trunc, _ = env.step(m(state))
  # env.close()