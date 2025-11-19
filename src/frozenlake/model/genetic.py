import numpy as np
from typing import Unpack
import gymnasium as gym
from gymnasium import Env
from .model import Model, Parameters

class Genetic(Model):
  def __init__(self, state_size: int, action_size: int, population_size: int, mutation_rate: float, **params: Unpack[Parameters]):
    self.population_size, self.mutation_rate = population_size, mutation_rate
    super().__init__(state_size, action_size, **params)

  def reset(self):
    super().reset()
    self.pop = np.stack([self._random_q() for _ in range(self.population_size)])

  def _random_q(self) -> np.ndarray: return np.random.randint(self.action_size, size=self.state_size).astype("uint8")
  def _fitness(self, env: Env) -> np.ndarray: return np.array([self._fitness_q(env, 100, q) for q in self.pop])

  # TODO: mostly duplicate code with self.test()
  def _fitness_q(self, env: Env, episodes: int, q: np.ndarray) -> float:
    success = 0
    for _ in range(episodes):
      state, _ = env.reset()
      done, trunc, reward = False, False, 0
      while not (done or trunc): state, reward, done, trunc, _ = env.step(q[state])
      success += float(reward)
    return success / episodes

  def _crossover(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    p = np.random.randint(1, len(q1))
    return np.concat((q1[:p], q2[p:]))

  def _mutate(self, q: np.ndarray) -> np.ndarray:
    q = q.copy()
    for s in range(q.shape[0]):
      if np.random.uniform(0, 1) < self.mutation_rate:
        q[s] = np.random.choice(self.action_size)
    return q

  def __call__(self, state: int, *, training: bool = False) -> int: return self.q[state]

  def train(self, env: Env, max_iter: int, threshold: float = 1e-9) -> int:
    generation = 0
    while generation < max_iter:
      fitness = self._fitness(env)
      half = self.population_size // 2
      best_half = fitness.argsort()[-half:]
      best = best_half[-1]

      # TODO: allow best (success rate) to stop train loop
      # if fitness[best] > threshold:
      #   break

      # keep the best individual (elitism)
      new_pop = [self.pop[best]]

      parents = self.pop[best_half]

      while len(new_pop) < self.population_size:
        p = parents[np.random.randint(0, len(parents), 2)]
        new_pop.append(self._mutate(self._crossover(*p)))
      self.pop = np.stack(new_pop)
      generation += 1

      print(f"gen: {generation}; best_idx: {best} -> {self.pop[best]}")

    self.q = self.pop[self._fitness(env).argmax()]

    return generation