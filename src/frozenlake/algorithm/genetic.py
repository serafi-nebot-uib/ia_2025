import time
import math
import numpy as np
from gymnasium import Env
from frozenlake.algorithm import Algorithm

class Genetic(Algorithm):
  def __init__(self, state_size: int, action_size: int, population_size: int, selection_pressure: float, mutation_rate: float, culling_rate: float):
    super().__init__(state_size, action_size)
    self.population_size, self.selection_pressure, self.mutation_rate, self.culling_rate = population_size, selection_pressure, mutation_rate, culling_rate

    # number of individuals in the population to select for crossover
    self.parents_size = math.ceil(self.population_size * self.selection_pressure)

    # number of best individuals to keep for the next generation
    self.best_size = math.ceil(self.population_size * self.culling_rate)

    # randomly create the first generation of the population
    self.pop = np.stack([self.random_policy() for _ in range(self.population_size)])

    # policy; state -> action table
    self.policy = np.zeros(self.state_size, dtype="uint8")

  def action(self, state: int, greedy: bool = True) -> int:
    """Choose which action to perform for the given `state` (`greedy` does nothing)"""
    return self.policy[state]

  def random_policy(self) -> np.ndarray:
    """Generate a completely random policy"""
    return np.random.randint(self.action_size, size=self.state_size).astype("uint8")

  def crossover(self, pa: np.ndarray, pb: np.ndarray) -> np.ndarray:
    """Randomly merge the genes of `pa` and `pb` with equal probability"""
    return np.where(np.random.uniform(size=len(pa)) < 0.5, pa, pb)

  def mutate(self, p: np.ndarray) -> np.ndarray:
    """Randomly change some genes of `p` with some low probability"""
    return np.where(np.random.uniform(size=len(p)) >= self.mutation_rate, p, self.random_policy())

  def fitness_policy(self, env: Env, p: np.ndarray, num_iter: int) -> tuple[float, float]:
    """
    Evaluate the policy of `p` for `num_iter` of iterations.

    Return the average number of steps and success rate.
    """
    s = 0.0
    steps = []
    for _ in range(num_iter):
      state, _ = env.reset()
      done, trunc, reward = False, False, 0
      step = 0
      while not (done or trunc):
        state, reward, done, trunc, _ = env.step(p[state]) # get action for current state and pass it to step()
        step += 1
      s += float(reward)
      steps.append(step)
    return np.mean(steps).item(), s / num_iter

  def fitness(self, env: Env) -> tuple:
    """Evaluate the policy of all the individuals in the population for 100 iterations"""
    return tuple(map(np.array, zip(*(self.fitness_policy(env, p, 100) for p in self.pop))))

  def train(self, env: Env, num_iter: int) -> int:
    iter = 0
    while iter < num_iter:
      iter_start = time.perf_counter()

      steps, fitness = self.fitness(env) # evaluate the fitness of the entire population
      best_pop = fitness.argsort()[-self.parents_size:]

      new_pop = list(self.pop[best_pop[-self.best_size:]]) # keep the best individuals (elitism)
      parents = self.pop[best_pop] # select the best individuals for crossover

      # randomly select two individuals from the best population and cross them
      # to generate the remaining part of the population
      while len(new_pop) < self.population_size:
        p = parents[np.random.randint(0, len(parents), 2)]
        new_pop.append(self.mutate(self.crossover(*p)))
      self.pop = np.stack(new_pop)

      iter += 1
      iter_end = time.perf_counter()
      # track statistics; survival rate is the one from the best individual
      self.train_stats(iter, np.mean(steps), fitness[best_pop[-1]].item(), iter_end - iter_start, sample=1)

    # set the best individual as the policy
    self.policy = self.pop[self.fitness(env)[1].argmax()]

    return iter