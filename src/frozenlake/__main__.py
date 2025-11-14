from frozenlake.model import Parameters, SARSA, QLearning, DoubleQLearning
import frozenlake.plot as plot
import numpy as np
import gymnasium as gym
from tqdm import tqdm
import matplotlib.pyplot as plt
import random

def test(model, *, episodes, seed: int | None = None):
  slippery = True
  env = gym.make("FrozenLake-v1", is_slippery=slippery, render_mode=None)
  episodes = model.train(env, episodes, seed=seed)
  # print(f"episodes: {episodes}")
  success = 0
  ntest = 100
  for _ in range(ntest):
    term, trunc = False, False
    state, _ = env.reset()
    while not (term or trunc): state, reward, term, trunc, _ = env.step(model(state))
    success += reward
  env.close()
  return success / ntest, episodes

if __name__ == "__main__":
  params = Parameters(state_size=4*4, action_size=4,
                      learning_rate=0.01, discount_rate=0.95,
                      eps=1.00, eps_min=0.10, eps_decay=0.995)
  m = QLearning(**params)
  sr, ep = test(m, episodes=40000)
  print(f"sr: {sr:.4f}; ep: {ep}")

  plt.plot(range(len(m.rewards_mean)), m.rewards_mean)
  plt.show()

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
  # ----- for SARSA/QLearning
  # plot.qtable(m.q, 4, 4)
  # ----- for DoubleQLearning
  # plot.qtable(m.qa, 4, 4)
  # plot.qtable(m.qb, 4, 4)

  # env = gym.make("FrozenLake-v1", is_slippery=slippery, render_mode="human")
  # state, _ = env.reset()
  # term, trunc = False, False
  # while not (term or trunc): state, _ , term, trunc, _ = env.step(m(state))
  # env.close()