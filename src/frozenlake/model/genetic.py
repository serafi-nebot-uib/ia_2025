import numpy as np
from typing import Unpack
from gymnasium import Env
from .model import Model, Parameters

class Genetic(Model):
  def __init__(self, state_size: int, action_size: int, population_size: int, mutation_rate: float, **params: Unpack[Parameters]):
    self.population_size, self.mutation_rate = population_size, mutation_rate
    super().__init__(state_size, action_size, **params)

  def reset(self):
    super().reset()
    self.pop = np.stack([self.random_policy() for _ in range(self.population_size)])

  def random_policy(self) -> np.ndarray: return np.random.randint(self.action_size, size=self.state_size).astype("uint8")
  def fitness_policy(self, env: Env, p: np.ndarray, num_iter: int) -> float: return self.test(env, num_iter, lambda s: p[s])
  def fitness(self, env: Env) -> np.ndarray: return np.array([self.fitness_policy(env, p, 100) for p in self.pop])
  def crossover(self, pa: np.ndarray, pb: np.ndarray) -> np.ndarray: return np.where(np.random.uniform(size=len(pa)) < 0.5, pa, pb)
  def mutate(self, p: np.ndarray) -> np.ndarray: return np.where(np.random.uniform(size=len(p)) < self.mutation_rate, self.random_policy(), p)

  def train(self, env: Env, max_iter: int, threshold: float = 1e-9) -> int:
    generation = 0
    while generation < max_iter:
      fitness = self.fitness(env)
      half = self.population_size // 2
      best_half = fitness.argsort()[-half:]
      best = best_half[-1]

      new_pop = [self.pop[best]] # keep the best individual (elitism)
      parents = self.pop[best_half]

      while len(new_pop) < self.population_size:
        p = parents[np.random.randint(0, len(parents), 2)]
        new_pop.append(self.mutate(self.crossover(*p)))
      self.pop = np.stack(new_pop)
      generation += 1

      print(f"gen: {generation:>3d}; best_idx: {best:>3d} -> {self.pop[best]}")

    self.policy = self.pop[self.fitness(env).argmax()]

    return generation