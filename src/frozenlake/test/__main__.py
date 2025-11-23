import gymnasium as gym
import matplotlib.pyplot as plt
from frozenlake.test import schedule_test, plot
from frozenlake.algorithm import SARSA, QLearning, MonteCarlo, DynamicProgramming, Genetic

SLIPPERY = True

env_conf = { "id": "FrozenLake-v1", "is_slippery": SLIPPERY, "render_mode": None }
env = gym.make(**env_conf)
DESC = env.unwrapped.desc # type: ignore[attr-defined]
PROBS = env.unwrapped.P # type: ignore[attr-defined]
STATE_SIZE = env.observation_space.n # type: ignore[attr-defined]
ACTION_SIZE = env.action_space.n # type: ignore[attr-defined]
env.close()

LEARN_TRAIN_ITER = 20000
LEARN_TEST_ITER = 1000
LEARN_AGGREGATION_WINDOW = 1000

LEARN = [
  (None, MonteCarlo,       { "state_size": STATE_SIZE, "action_size": ACTION_SIZE,             "dr": 0.99, "er": 1.00, "er_min": 0.01, "er_decay": 0.9995 }, LEARN_TRAIN_ITER, LEARN_TEST_ITER),
  (None, SARSA,            { "state_size": STATE_SIZE, "action_size": ACTION_SIZE, "lr": 0.10, "dr": 0.99, "er": 1.00, "er_min": 0.01, "er_decay": 0.9995 }, LEARN_TRAIN_ITER, LEARN_TEST_ITER),
  (None, QLearning,        { "state_size": STATE_SIZE, "action_size": ACTION_SIZE, "lr": 0.10, "dr": 0.99, "er": 1.00, "er_min": 0.01, "er_decay": 0.9995 }, LEARN_TRAIN_ITER, LEARN_TEST_ITER),
]

SEARCH_TRAIN_ITER = 20
SEARCH_TEST_ITER = 1000

SEARCH = [
  (None, Genetic, { "state_size": STATE_SIZE, "action_size": ACTION_SIZE, "population_size": 100, "selection_pressure": 0.50, "mutation_rate": 0.08, "culling_rate": 1/100 }, SEARCH_TRAIN_ITER, SEARCH_TEST_ITER),
  (None, DynamicProgramming, { "state_size": STATE_SIZE, "action_size": ACTION_SIZE, "probs": PROBS, "dr": 0.99 }, SEARCH_TRAIN_ITER, SEARCH_TEST_ITER),
]

if __name__ == "__main__":
  learn_data = schedule_test(LEARN, env_conf)
  search_data = schedule_test(SEARCH, env_conf)

  data = learn_data | search_data

  if len(learn_data) > 0:
    plot.train_perf(learn_data, LEARN_AGGREGATION_WINDOW)
    plot.train_time(learn_data, LEARN_AGGREGATION_WINDOW)
    plot.train_steps(learn_data, LEARN_AGGREGATION_WINDOW)
    plot.train_actions(learn_data)
    plot.train_actions_history(learn_data)

  if len(search_data) > 0:
    plot.train_perf(search_data)
    plot.train_time(search_data)
    plot.train_steps(search_data)

  data = learn_data | search_data

  if len(data) > 0:
    plot.test_steps(data)
    plot.qtable(data, DESC)

  plt.show()