import numpy as np
from gymnasium import Env
from frozenlake.const import DEBUG, STATS, SAMPLE, RAND_TIE_BREAK

class Algorithm:
  def __init__(self, state_size: int, action_size: int):
    self.name = self.__class__.__name__
    self.state_size, self.action_size = state_size, action_size
    # data structure for statistics, only used when STATS > 0
    self.time, self.performance, self.steps, self.actions_history = [], [], [], []
    self.actions = np.zeros(self.action_size, dtype=np.int64)

  def action(self, state: int, greedy: bool = True) -> int:
    """Choose which action to perform for the given `state` (`greedy=False` for epsilon-greedy)"""
    raise NotImplementedError()

  def train(self, env: Env, num_iter: int) -> int:
    """Train with the provided environment `env` for `num_iter` of iterations"""
    raise NotImplementedError()

  def train_stats(self, iter: int, steps: float, perf: float, t: float, sample: int = SAMPLE):
    """Save the statistics for the current iteration of the train loop"""
    if DEBUG > 0 or STATS > 0:
      self.steps.append(steps)
      self.performance.append(perf)
      self.time.append(t)
      self.actions_history.append(self.actions.copy())

    # show the statistics for the last sample amount of iterations
    if DEBUG > 0 and (not sample or iter % sample == 0):
      perf_avg = np.mean(self.performance[-sample:])
      time_avg = np.mean(self.time[-sample:])
      time_total = np.sum(self.time[-sample:])
      print(f"{self.name:<16} {iter:>7d} | perf: {perf_avg:>4.2f} | t: {time_total:.6f} s", end="")
      if sample > 1: print(f" (total) {time_avg:.6f} s (avg)", end="")
      print(flush=True)

  def run(self, env: Env, greedy: bool = True) -> tuple[int, float]:
    """
    Run the environment `env` once (`greedy=False` for epsilon-greedy).

    Return the number of steps and the total accumulated reward for the episode.
    """
    state, _ = env.reset() # restart to get the initial state
    steps, reward_total = 0, 0
    done, trunc = False, False
    while not (done or trunc):
      state, reward, done, trunc, _ = env.step(self.action(state, greedy=greedy)) # get next action and pass it to environment
      reward_total += float(reward)
      steps += 1
    return steps, reward_total

  def test(self, env: Env, num_iter: int, greedy: bool = True) -> tuple[list[int], list[int]]:
    """
    Test the environment `env` for `num_iter` times (`greedy=False` for epsilon-greedy).

    Return the number of steps and whether the goal was reached for each episode.
    """
    steps, success = [], []
    for _ in range(num_iter):
      step, reward = self.run(env, greedy)
      steps.append(step)
      success.append(int(reward > 0)) # 1 on episode success, 0 otherwise
    return steps, success

class LearningAlgorithm(Algorithm):
  def __init__(self, state_size: int, action_size: int, lr: float, dr: float, er: float, er_min: float, er_decay: float):
    super().__init__(state_size, action_size)
    self.lr, self.dr = lr, dr
    self.er, self.er_min, self.er_decay = er, er_min, er_decay
    # initialize the Q table (state-action table) with all zeros
    self.q = np.zeros((self.state_size, self.action_size), dtype="float32")

  def action(self, state: int, greedy: bool = True, tiebreak: bool = RAND_TIE_BREAK) -> int:
    """Choose which action to perform for the given `state` (`greedy=False` for epsilon-greedy; `tiebreak=True` for uniform argmax selection)."""
    if not greedy and np.random.rand() < self.er:
      return np.random.choice(self.action_size) # randomly sample an action

    if tiebreak:
      # randomly choose among the maximum indices
      return np.random.choice(np.flatnonzero(np.isclose(self.q[state], self.q[state].max())))
    else:
      # choose the first maximum index
      return self.q[state].argmax()