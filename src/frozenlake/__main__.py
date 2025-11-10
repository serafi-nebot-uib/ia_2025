from frozenlake.model import SARSA, QLearning, DoubleQLearning
import frozenlake.plot as plot
import gymnasium as gym

if __name__ == "__main__":
  slippery = True
  env = gym.make("FrozenLake-v1", is_slippery=slippery, render_mode=None)

  # m = SARSA(env.observation_space.n, env.action_space.n, learning_rate=0.10, gamma=0.95)
  # m = QLearning(env.observation_space.n, env.action_space.n, learning_rate=0.10, gamma=0.95)
  m = DoubleQLearning(env.observation_space.n, env.action_space.n, learning_rate=0.10, gamma=0.95)
  m.train(env, 10000)

  success = 0
  ntest = 10000
  for _ in range(ntest):
    term, trunc = False, False
    state, _ = env.reset()
    while not (term or trunc): state, reward, term, trunc, _ = env.step(m(state))
    success += reward
  print(f"success rate: {success / ntest:.4f}")
  env.close()

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