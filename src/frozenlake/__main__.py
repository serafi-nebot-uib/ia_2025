import random
import numpy as np
import gymnasium as gym
from gymnasium import Env
import matplotlib.pyplot as plt
import frozenlake.plot as plot
from frozenlake.algorithm import Algorithm, SARSA, QLearning, DoubleQLearning, MonteCarlo, DynamicProgramming, Genetic

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

def test(alg: Algorithm, *, num_iter, seed: int | None = None):
  env = gym.make("FrozenLake-v1", is_slippery=SLIPPERY, render_mode=None)

  if seed:
    np.random.seed(seed)
    random.seed(seed)

  env.reset(seed=seed)
  num_iter = alg.train(env, num_iter)
  sr = alg.test(env, 1000)
  env.close()

  return sr, num_iter

if __name__ == "__main__":
  env = gym.make("FrozenLake-v1", is_slippery=SLIPPERY, render_mode=None)
  desc = env.unwrapped.desc
  probs = env.unwrapped.P
  state_size = env.observation_space.n
  action_size = env.action_space.n
  env.close()

  # algs = [
  #   Genetic(state_size, action_size, population_size=100, mutation_rate=0.08),
  #   DynamicProgramming(state_size, action_size, probs, 0.95)
  # ]
  # data_sr, data_t = {}, {}
  # for alg in algs:
  #   sr, ep = test(alg, num_iter=40)
  #   print(f"sr: {sr:.4f}; ep: {ep}")
  #   data_sr[alg.__class__.__name__] = alg.performance
  #   data_t[alg.__class__.__name__] = alg.time

  # fig = plot.success_rate(data_sr)
  # fig = plot.train_time(data_t)
  # plt.show()

  algs = [
         MonteCarlo(state_size, action_size,          dr=0.99, er=1.00, er_min=0.01, er_decay=0.9995),
              SARSA(state_size, action_size, lr=0.20, dr=0.95, er=1.00, er_min=0.01, er_decay=0.9995),
          QLearning(state_size, action_size, lr=0.20, dr=0.95, er=1.00, er_min=0.01, er_decay=0.9995),
    DoubleQLearning(state_size, action_size, lr=0.20, dr=0.95, er=1.00, er_min=0.01, er_decay=0.9995)
  ]

  data_sr, data_t = {}, {}
  for alg in algs:
    sr, ep = test(alg, num_iter=5000)
    print(f"sr: {sr:.4f}; ep: {ep}")
    data_sr[alg.name] = alg.performance
    data_t[alg.name] = alg.time

  # fig = plot.success_rate(data_sr, 1000)
  # fig = plot.train_time(data_t, 1000)
  # plt.show()
  for alg in algs: plot.qtable(alg.name, alg.q, desc)
  plt.show()

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