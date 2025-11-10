from frozenlake.model import SARSA, QLearning
import frozenlake.plot as plot
import gymnasium as gym

if __name__ == "__main__":
  slippery = True
  env = gym.make("FrozenLake-v1", is_slippery=slippery, render_mode=None)

  # m = SARSA(env.observation_space.n, env.action_space.n, learning_rate=0.10, gamma=0.998, epsilon=0.1)
  m = QLearning(env.observation_space.n, env.action_space.n, learning_rate=0.10, gamma=0.998, epsilon=0.1)
  m.train(env, 20000)

  # view Q table (policy)
  # requirements:
  #     - uv pip install matplotlib
  plot.qtable(m.q, 4, 4)

  env = gym.make("FrozenLake-v1", is_slippery=slippery, render_mode="human")
  state, _ = env.reset()
  term, trunc = False, False
  while not (term or trunc): state, _ , term, trunc, _ = env.step(m(state))