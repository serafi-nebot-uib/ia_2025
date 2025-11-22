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
  (None, MonteCarlo,       { "state_size": STATE_SIZE, "action_size": ACTION_SIZE,             "dr": 0.95, "er": 1.00, "er_min": 0.01, "er_decay": 0.9995 }, LEARN_TRAIN_ITER, LEARN_TEST_ITER),
  (None, SARSA,            { "state_size": STATE_SIZE, "action_size": ACTION_SIZE, "lr": 0.10, "dr": 0.99, "er": 1.00, "er_min": 0.01, "er_decay": 0.9995 }, LEARN_TRAIN_ITER, LEARN_TEST_ITER),
  (None, QLearning,        { "state_size": STATE_SIZE, "action_size": ACTION_SIZE, "lr": 0.10, "dr": 0.99, "er": 1.00, "er_min": 0.01, "er_decay": 0.9995 }, LEARN_TRAIN_ITER, LEARN_TEST_ITER),
  # (None, AverageQLearning, { "state_size": STATE_SIZE, "action_size": ACTION_SIZE, "lr": 0.10, "dr": 0.99, "er": 1.00, "er_min": 0.01, "er_decay": 0.9995 }, LEARN_TRAIN_ITER, LEARN_TEST_ITER),
]

SEARCH_TRAIN_ITER = 20
SEARCH_TEST_ITER = 1000

SEARCH = [
  (None, Genetic, { "state_size": STATE_SIZE, "action_size": ACTION_SIZE, "population_size": 100, "selection_pressure": 0.50, "mutation_rate": 0.08, "culling_rate": 1/100 }, SEARCH_TRAIN_ITER, SEARCH_TEST_ITER),
  (None, DynamicProgramming, { "state_size": STATE_SIZE, "action_size": ACTION_SIZE, "probs": PROBS, "dr": 0.99 }, SEARCH_TRAIN_ITER, SEARCH_TEST_ITER),
]

if __name__ == "__main__":
  # data = test(LEARN)
  data = test(SEARCH)

  # for name in data:
  #   steps = data[name]["test"]["steps"]
  #   perf = data[name]["test"]["perf"]
  #   print(f"{name} | steps: {np.mean(steps):6.4f} | perf: {np.mean(perf):6.4f}")

  plot.train_perf(data, 1)
  plot.train_time(data, 1)
  plot.train_steps(data, 1)
  plot.train_actions(data)
  # plot.train_actions_history(data, 1000)
  plot.test_steps(data)
  # plot.qtable(data, DESC)
  plt.show()

  # env = gym.make("FrozenLake-v1", is_slippery=SLIPPERY, render_mode="human")
  # state, _ = env.reset()
  # term, trunc = False, False
  # while not (term or trunc): state, _ , term, trunc, _ = env.step(m(state))
  # env.close()