from frozenlake.model import Parameters, Model, SARSA, QLearning, DoubleQLearning, MonteCarlo, DynamicProgramming
import frozenlake.plot as plot
import numpy as np
import gymnasium as gym
from tqdm import tqdm
import matplotlib.pyplot as plt
import random

SLIPPERY = True

def test(model: Model, *, episodes, seed: int | None = None):
  env = gym.make("FrozenLake-v1", is_slippery=SLIPPERY, render_mode=None)

  if seed:
    np.random.seed(seed)
    random.seed(seed)

  env.reset(seed=seed)
  episodes = model.train(env, episodes)
  # print(f"episodes: {episodes}")
  success = 0
  ntest = 1000
  for _ in range(ntest):
    term, trunc = False, False
    state, _ = env.reset(seed=seed)
    while not (term or trunc): state, reward, term, trunc, _ = env.step(model(state))
    success += reward

  m.printPolicy(env)

  env.close()
  return success / ntest, episodes

if __name__ == "__main__":
  # TODO: is there a better way to retrieve/build P without making a new environment just for it?
  env = gym.make("FrozenLake-v1", is_slippery=SLIPPERY, render_mode=None)
  probs = env.unwrapped.P
  env.close()

  params = Parameters(lr=0.20,
                      dr=0.95,
                      er=1.00, er_min=0.01, er_decay=0.995)
  # m = QLearning(state_size=4*4, action_size=4, **params)
  # m = SARSA(state_size=4*4, action_size=4, **params)
  # m = MonteCarlo(state_size=4*4, action_size=4, **params)
  m = DynamicProgramming(state_size=4*4, action_size=4, probs=probs, **params)
  sr, ep = test(m, episodes=40000)
  print(f"sr: {sr:.4f}; ep: {ep}")

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

  env = gym.make("FrozenLake-v1", is_slippery=SLIPPERY, render_mode="human")
  state, _ = env.reset()
  term, trunc = False, False
  while not (term or trunc): state, _ , term, trunc, _ = env.step(m(state))
  env.close()