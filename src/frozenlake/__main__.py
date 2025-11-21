import random
import numpy as np
import gymnasium as gym
from gymnasium import Env
import matplotlib.pyplot as plt
import frozenlake.plot as plot
from frozenlake.algorithm import Algorithm, SARSA, QLearning, AverageQLearning, MonteCarlo, DynamicProgramming, Genetic

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
  (MonteCarlo,       { "state_size": STATE_SIZE, "action_size": ACTION_SIZE,             "dr": 0.99, "er": 1.00, "er_min": 0.01, "er_decay": 0.9995 }, LEARN_TRAIN_ITER, LEARN_TEST_ITER),
  (SARSA,            { "state_size": STATE_SIZE, "action_size": ACTION_SIZE, "lr": 0.20, "dr": 0.99, "er": 1.00, "er_min": 0.01, "er_decay": 0.9995 }, LEARN_TRAIN_ITER, LEARN_TEST_ITER),
  (QLearning,        { "state_size": STATE_SIZE, "action_size": ACTION_SIZE, "lr": 0.20, "dr": 0.99, "er": 1.00, "er_min": 0.01, "er_decay": 0.9995 }, LEARN_TRAIN_ITER, LEARN_TEST_ITER),
  (AverageQLearning, { "state_size": STATE_SIZE, "action_size": ACTION_SIZE, "lr": 0.20, "dr": 0.99, "er": 1.00, "er_min": 0.01, "er_decay": 0.9995 }, LEARN_TRAIN_ITER, LEARN_TEST_ITER),
]

SEARCH_TRAIN_ITER = 20
SEARCH_TEST_ITER = 1000

SEARCH = [
  (Genetic, { "state_size": STATE_SIZE, "action_size": ACTION_SIZE, "population_size": 100, "selection_pressure": 0.50, "mutation_rate": 0.08, "culling_rate": 1/100 }, SEARCH_TRAIN_ITER, SEARCH_TEST_ITER),
  (DynamicProgramming, { "state_size": STATE_SIZE, "action_size": ACTION_SIZE, "probs": PROBS, "dr": 0.95 }, SEARCH_TRAIN_ITER, SEARCH_TEST_ITER),
]

def test(config: list[tuple[type[Algorithm], dict, int, int]]):
  env = gym.make("FrozenLake-v1", is_slippery=SLIPPERY, render_mode=None)

  data = {}
  for alg, params, train_iter, test_iter in config:
    a = alg(**params)

    a.train(env, train_iter)
    steps, success = a.test(env, test_iter)
    data[a.name] = {
      "alg": a,
      "train": { "steps": a.steps, "actions": a.actions, "perf": a.performance, "time": a.time },
      "test": { "steps": steps, "perf": success }
    }

  env.close()

  return data

if __name__ == "__main__":
  learn_data = test(LEARN)
  # search_data = test(SEARCH)

  data = learn_data
  # plot.success_rate(data, 1000)
  # plot.train_time(data, 1000)
  # plot.train_steps(data, 1000)
  # plot.test_steps(learn_data | search_data, "test", "steps")
  plot.train_actions(data)
  # for alg in learn_data.values(): plot.qtable(alg["alg"], DESC)
  plt.show()

  # env = gym.make("FrozenLake-v1", is_slippery=SLIPPERY, render_mode="human")
  # state, _ = env.reset()
  # term, trunc = False, False
  # while not (term or trunc): state, _ , term, trunc, _ = env.step(m(state))
  # env.close()