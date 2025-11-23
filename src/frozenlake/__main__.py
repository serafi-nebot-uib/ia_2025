import gymnasium as gym
import numpy as np
from frozenlake.algorithm import LearningAlgorithm, SARSA, QLearning, MonteCarlo, DynamicProgramming, Genetic

SLIPPERY = True

env_conf = { "id": "FrozenLake-v1", "is_slippery": SLIPPERY, "render_mode": None }
env = gym.make(**env_conf)
PROBS = env.unwrapped.P # type: ignore[attr-defined]
STATE_SIZE = env.observation_space.n # type: ignore[attr-defined]
ACTION_SIZE = env.action_space.n # type: ignore[attr-defined]
env.close()

if __name__ == "__main__":
  alg = MonteCarlo(state_size=STATE_SIZE, action_size=ACTION_SIZE,          dr=0.99, er=1.00, er_min=0.01, er_decay=0.9995)
  # alg =      SARSA(state_size=STATE_SIZE, action_size=ACTION_SIZE, lr=0.10, dr=0.99, er=1.00, er_min=0.01, er_decay=0.9995)
  # alg =  QLearning(state_size=STATE_SIZE, action_size=ACTION_SIZE, lr=0.10, dr=0.99, er=1.00, er_min=0.01, er_decay=0.9995)
  # alg = Genetic(state_size=STATE_SIZE, action_size=ACTION_SIZE, population_size=100, selection_pressure=0.50, mutation_rate=0.08, culling_rate=1/100)
  # alg = DynamicProgramming(state_size=STATE_SIZE, action_size=ACTION_SIZE, probs=PROBS, dr=0.99)

  train_iter = 20000 if isinstance(alg, LearningAlgorithm) else 20
  test_iter = 1000

  # train the algorithm for trai_iter cycles
  alg.train(env, train_iter)
  # test the policy test_iter episodes to calculate the average number of steps & survival rate
  steps, success = alg.test(env, test_iter)
  print(f"{alg.name} | {test_iter} episodes -> avg steps: {np.mean(steps):6.4f}; sr: {np.mean(success):6.4f}")

  # change the render mode to human so that we get the animation
  env = gym.make(**(env_conf | { "render_mode": "human" }))

  # show as many animated episodes as the user wants to
  try:
    done = False
    while not done:
      nstep, reward = alg.run(env) # execute the environment for 1 episode
      print(f"{alg.name} | steps: {nstep} -> {'success' if reward > 0 else 'fail'}")
      # ask the user if another episode should be executed
      print("press enter to run another episode, \"quit\" or \"q\" to exit")
      print("> ", end="")
      done = input() in ("quit", "q")
  except KeyboardInterrupt:
    pass

  env.close()