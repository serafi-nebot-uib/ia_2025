import random
import json
from pathlib import Path
from multiprocessing import Pool
import numpy as np
import gymnasium as gym
from gymnasium import Env
import matplotlib.pyplot as plt
import frozenlake.plot as plot
from frozenlake.algorithm import Algorithm, SARSA, QLearning, AverageQLearning, MonteCarlo, DynamicProgramming, Genetic
from frozenlake.test.test import test

SLIPPERY = True

env = gym.make("FrozenLake-v1", is_slippery=SLIPPERY, render_mode=None)
DESC = env.unwrapped.desc
PROBS = env.unwrapped.P
STATE_SIZE = env.observation_space.n
ACTION_SIZE = env.action_space.n
env.close()

LEARN_TRAIN_ITER = 20000
LEARN_TEST_ITER = 1000

LEARN = [
  (None, MonteCarlo,       { "state_size": STATE_SIZE, "action_size": ACTION_SIZE,             "dr": 0.99, "er": 1.00, "er_min": 0.01, "er_decay": 0.9995 }, LEARN_TRAIN_ITER, LEARN_TEST_ITER),
  (None, SARSA,            { "state_size": STATE_SIZE, "action_size": ACTION_SIZE, "lr": 0.01, "dr": 0.99, "er": 1.00, "er_min": 0.01, "er_decay": 0.9995 }, LEARN_TRAIN_ITER, LEARN_TEST_ITER),
  (None, QLearning,        { "state_size": STATE_SIZE, "action_size": ACTION_SIZE, "lr": 0.01, "dr": 0.99, "er": 1.00, "er_min": 0.01, "er_decay": 0.9995 }, LEARN_TRAIN_ITER, LEARN_TEST_ITER),
  # (AverageQLearning, { "state_size": STATE_SIZE, "action_size": ACTION_SIZE, "lr": 0.01, "dr": 0.99, "er": 1.00, "er_min": 0.01, "er_decay": 0.9995 }, LEARN_TRAIN_ITER, LEARN_TEST_ITER),
]

SEARCH_TRAIN_ITER = 20
SEARCH_TEST_ITER = 1000

SEARCH = [
  (Genetic, { "state_size": STATE_SIZE, "action_size": ACTION_SIZE, "population_size": 100, "selection_pressure": 0.50, "mutation_rate": 0.08, "culling_rate": 1/100 }, SEARCH_TRAIN_ITER, SEARCH_TEST_ITER),
  (DynamicProgramming, { "state_size": STATE_SIZE, "action_size": ACTION_SIZE, "probs": PROBS, "dr": 0.95 }, SEARCH_TRAIN_ITER, SEARCH_TEST_ITER),
]

if __name__ == "__main__":
  config = [
    ("dr=0.10", MonteCarlo, { "state_size": STATE_SIZE, "action_size": ACTION_SIZE,             "dr": 0.10, "er": 1.00, "er_min": 0.01, "er_decay": 0.9995 }, LEARN_TRAIN_ITER, LEARN_TEST_ITER),
    ("dr=0.20", MonteCarlo, { "state_size": STATE_SIZE, "action_size": ACTION_SIZE,             "dr": 0.20, "er": 1.00, "er_min": 0.01, "er_decay": 0.9995 }, LEARN_TRAIN_ITER, LEARN_TEST_ITER),
    ("dr=0.30", MonteCarlo, { "state_size": STATE_SIZE, "action_size": ACTION_SIZE,             "dr": 0.30, "er": 1.00, "er_min": 0.01, "er_decay": 0.9995 }, LEARN_TRAIN_ITER, LEARN_TEST_ITER),
    ("dr=0.40", MonteCarlo, { "state_size": STATE_SIZE, "action_size": ACTION_SIZE,             "dr": 0.40, "er": 1.00, "er_min": 0.01, "er_decay": 0.9995 }, LEARN_TRAIN_ITER, LEARN_TEST_ITER),
    ("dr=0.50", MonteCarlo, { "state_size": STATE_SIZE, "action_size": ACTION_SIZE,             "dr": 0.50, "er": 1.00, "er_min": 0.01, "er_decay": 0.9995 }, LEARN_TRAIN_ITER, LEARN_TEST_ITER),
    ("dr=0.60", MonteCarlo, { "state_size": STATE_SIZE, "action_size": ACTION_SIZE,             "dr": 0.60, "er": 1.00, "er_min": 0.01, "er_decay": 0.9995 }, LEARN_TRAIN_ITER, LEARN_TEST_ITER),
    ("dr=0.70", MonteCarlo, { "state_size": STATE_SIZE, "action_size": ACTION_SIZE,             "dr": 0.70, "er": 1.00, "er_min": 0.01, "er_decay": 0.9995 }, LEARN_TRAIN_ITER, LEARN_TEST_ITER),
    ("dr=0.80", MonteCarlo, { "state_size": STATE_SIZE, "action_size": ACTION_SIZE,             "dr": 0.80, "er": 1.00, "er_min": 0.01, "er_decay": 0.9995 }, LEARN_TRAIN_ITER, LEARN_TEST_ITER),
    ("dr=0.90", MonteCarlo, { "state_size": STATE_SIZE, "action_size": ACTION_SIZE,             "dr": 0.90, "er": 1.00, "er_min": 0.01, "er_decay": 0.9995 }, LEARN_TRAIN_ITER, LEARN_TEST_ITER),
    ("dr=1.00", MonteCarlo, { "state_size": STATE_SIZE, "action_size": ACTION_SIZE,             "dr": 1.00, "er": 1.00, "er_min": 0.01, "er_decay": 0.9995 }, LEARN_TRAIN_ITER, LEARN_TEST_ITER),
  ]
  data = test(LEARN)

  for name in data:
    steps = data[name]["test"]["steps"]
    perf = data[name]["test"]["perf"]
    print(f"{name} | steps: {np.mean(steps):6.4f} | perf: {np.mean(perf):6.4f}")

  # figs = []
  # figs.append(plot.train_perf(data, 1000))
  # figs.append(plot.train_time(data, 1000))
  # figs.append(plot.train_steps(data, 1000))
  # plot.train_actions(data)
  # figs.append(plot.test_steps(data))
  # # for alg in learn_data.values(): plot.qtable(alg["alg"], DESC)
  # plt.show()

  # env = gym.make("FrozenLake-v1", is_slippery=SLIPPERY, render_mode="human")
  # state, _ = env.reset()
  # term, trunc = False, False
  # while not (term or trunc): state, _ , term, trunc, _ = env.step(m(state))
  # env.close()