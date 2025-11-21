import time
import math
import numpy as np
from gymnasium import Env
from frozenlake.algorithm import Algorithm

class Genetic(Algorithm):
  def __init__(self, state_size: int, action_size: int, population_size: int, selection_pressure: float, mutation_rate: float, culling_rate: float):
    super().__init__(state_size, action_size)
    self.population_size, self.selection_pressure, self.mutation_rate = population_size, selection_pressure, mutation_rate
    self.culling_rate = culling_rate
    self.parents_size = math.ceil(self.population_size * self.selection_pressure)
    self.best_size = math.ceil(self.population_size * self.culling_rate)
    self.pop = np.stack([self.random_policy() for _ in range(self.population_size)])
    self.policy = np.zeros(self.state_size, dtype="uint8")

  def action(self, state: int, greedy: bool = True) -> int: return self.policy[state]
  def random_policy(self) -> np.ndarray: return np.random.randint(self.action_size, size=self.state_size).astype("uint8")
  def crossover(self, pa: np.ndarray, pb: np.ndarray) -> np.ndarray: return np.where(np.random.uniform(size=len(pa)) < 0.5, pa, pb)
  def mutate(self, p: np.ndarray) -> np.ndarray: return np.where(np.random.uniform(size=len(p)) >= self.mutation_rate, p, self.random_policy())

  def fitness_policy(self, env: Env, p: np.ndarray, num_iter: int) -> tuple[float, float]:
    s = 0.0
    steps = []
    for _ in range(num_iter):
      state, _ = env.reset()
      done, trunc, reward = False, False, 0
      step = 0
      while not (done or trunc):
        state, reward, done, trunc, _ = env.step(p[state])
        step += 1
      s += float(reward)
      steps.append(step)
    return np.mean(steps).item(), s / num_iter

  def fitness(self, env: Env) -> tuple: return tuple(map(np.array, zip(*(self.fitness_policy(env, p, 100) for p in self.pop))))

  def train(self, env: Env, num_iter: int) -> int:
    iter = 0
    while iter < num_iter:
      iter_start = time.perf_counter()

      steps, fitness = self.fitness(env)
      best_pop = fitness.argsort()[-self.parents_size:]
      best = best_pop[-1]

      new_pop = list(self.pop[best_pop[-self.best_size:]]) # keep the best individuals (elitism)
      parents = self.pop[best_pop]

      while len(new_pop) < self.population_size:
        p = parents[np.random.randint(0, len(parents), 2)]
        new_pop.append(self.mutate(self.crossover(*p)))
      self.pop = np.stack(new_pop)

      iter += 1
      iter_end = time.perf_counter()
      self.train_stats(iter, np.mean(steps), fitness[best].item(), iter_end - iter_start, sample=1)

    self.policy = self.pop[self.fitness(env)[1].argmax()]

    return iter