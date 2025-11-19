import numpy as np
from gymnasium import Env
from frozenlake.model import Model
from frozenlake.const import DEBUG
import time

class Genetic(Model):
  def __init__(self, state_size: int, action_size: int, population_size: int, mutation_rate: float):
    super().__init__(state_size, action_size)
    self.population_size, self.mutation_rate = population_size, mutation_rate
    self.pop = np.stack([self.random_policy() for _ in range(self.population_size)])

  def random_policy(self) -> np.ndarray: return np.random.randint(self.action_size, size=self.state_size).astype("uint8")

  def fitness_policy(self, env: Env, p: np.ndarray, num_iter: int) -> float:
    s = 0.0
    for _ in range(num_iter):
      state, _ = env.reset()
      done, trunc, reward = False, False, 0
      while not (done or trunc): state, reward, done, trunc, _ = env.step(p[state])
      s += float(reward)
    return s / num_iter

  def fitness(self, env: Env) -> np.ndarray: return np.array([self.fitness_policy(env, p, 100) for p in self.pop])
  def crossover(self, pa: np.ndarray, pb: np.ndarray) -> np.ndarray: return np.where(np.random.uniform(size=len(pa)) < 0.5, pa, pb)
  def mutate(self, p: np.ndarray) -> np.ndarray: return np.where(np.random.uniform(size=len(p)) < self.mutation_rate, self.random_policy(), p)

  def train(self, env: Env, max_iter: int, threshold: float = 1e-9) -> int:
    generation = 0
    while generation < max_iter:
      iter_start = time.perf_counter()

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

      iter_end = time.perf_counter()

      if DEBUG > 0:
        print(f"{generation:>7d} | {fitness[best]:>4.2f} : {self.pop[best]} | {iter_end - iter_start:.6f} sec")

    self.policy = self.pop[self.fitness(env).argmax()]

    return generation