import numpy as np
from gymnasium import Env

class Model:
    def printOutput(self, env: Env):
        arrows = {0: '⇐', 1: '⇓', 2: '⇒', 3: '⇑'}; goal ="⊕"; wall =" " # visible wall:"▓"
        q = (self.qa + self.qb) / 2 if hasattr(self, "qa") else self.q
        v = np.max(q, axis=1)
        policy = np.argmax(q, axis=1)
        dim = int(np.sqrt(self.state_size))
        desc = env.unwrapped.desc
        print("       ╭────╮        ╭────╮   ")
        print("┍━━━━━━┥ v* ┝━━━━━┯━━┥ π* ┝━━┑")
        print("│      ╰────╯     │  ╰────╯  │")
        for y in range(dim):
            row_v = " ".join(f"{v[y*dim + x]:.1f}" for x in range(dim))
            row_pi = " ".join(wall if desc[y, x] == b'H' else goal if desc[y, x] == b'G' else arrows[policy[y*dim + x]] for x in range(dim))
            print(f"│ {row_v} │ {row_pi}  │")
        print("┕━━━━━━━━━━━━━━━━━┷━━━━━━━━━━┙")

class SARSA(Model):
  def __init__(self, state_size: int, action_size: int, learning_rate: float, gamma: float, eps: float = 1.0, eps_min: float = 0.10, eps_decay: float = 0.995):
    self.state_size, self.action_size = state_size, action_size
    self.learning_rate, self.gamma = learning_rate, gamma
    self.eps, self.eps_min, self.eps_decay = eps, eps_min, eps_decay
    self.reset()

  def reset(self): self.q = np.zeros((self.state_size, self.action_size), dtype="float32")

  def __call__(self, state: int, training: bool = False):
    if training and np.random.uniform(0, 1) < self.eps: return np.random.choice(self.action_size)
    else: return self.q[state].argmax()

  def update(self, state: int, action: int, reward: float, new_state: int, new_action: int, final: bool = False):
    target = reward + self.gamma * self.q[new_state, new_action] * (not final)
    self.q[state, action] += self.learning_rate * (target - self.q[state, action])

  def train(self, env: Env, episodes: int):
    for _ in range(episodes):
      state, _ = env.reset()
      action = self(state)
      term, trunc = False, False
      while not (term or trunc):
        new_state, reward, term, trunc, _ = env.step(action)
        new_action = self(new_state, True)
        self.update(state, action, reward, new_state, new_action, term)
        state, action = new_state, new_action
      self.eps = max(self.eps_min, self.eps * self.eps_decay)

class QLearning(Model):
  def __init__(self, state_size: int, action_size: int, learning_rate: float, gamma: float, eps: float = 1.0, eps_min: float = 0.10, eps_decay: float = 0.995):
    self.state_size, self.action_size = state_size, action_size
    self.learning_rate, self.gamma = learning_rate, gamma
    self.eps, self.eps_min, self.eps_decay = eps, eps_min, eps_decay
    self.reset()

  def reset(self): self.q = np.zeros((self.state_size, self.action_size), dtype="float32")

  def __call__(self, state: int, training: bool = False):
    if training and np.random.uniform(0, 1) < self.eps: return np.random.choice(self.action_size)
    else: return self.q[state].argmax()

  def update(self, state: int, action: int, reward: float, new_state: int, final: bool = False):
    target = reward + self.gamma * self.q[new_state].max() * (not final)
    self.q[state, action] += self.learning_rate * (target - self.q[state, action])

  def train(self, env: Env, episodes: int):
    for _ in range(episodes):
      state, _ = env.reset()
      term, trunc = False, False
      while not (term or trunc):
        action = self(state, True)
        new_state, reward, term, trunc, _ = env.step(action)
        self.update(state, action, reward, new_state, term)
        state = new_state
      self.eps = max(self.eps_min, self.eps * self.eps_decay)

class DoubleQLearning(Model):
  def __init__(self, state_size: int, action_size: int, learning_rate: float, gamma: float, eps: float = 1.0, eps_min: float = 0.10, eps_decay: float = 0.995):
    self.state_size, self.action_size = state_size, action_size
    self.learning_rate, self.gamma = learning_rate, gamma
    self.eps, self.eps_min, self.eps_decay = eps, eps_min, eps_decay
    self.reset()

  def reset(self):
    self.qa = np.zeros((self.state_size, self.action_size), dtype="float32")
    self.qb = np.zeros((self.state_size, self.action_size), dtype="float32")

  def __call__(self, state: int, training: bool = False):
    if training and np.random.uniform(0, 1) < self.eps: return np.random.choice(self.action_size)
    else: return np.argmax((self.qa[state] + self.qb[state]) / 2)

  def update(self, state: int, action: int, reward: float, new_state: int, final: bool = False):
    if np.random.uniform(0, 1) < 0.5:
      target = reward + self.gamma * self.qb[new_state].max() * (not final)
      td_error = target - self.qa[state, action]
      self.qa[state, action] += self.learning_rate * td_error
    else:
      target = reward + self.gamma * self.qa[new_state].max() * (not final)
      td_error = target - self.qb[state, action]
      self.qb[state, action] += self.learning_rate * td_error

  def train(self, env: Env, episodes: int):
    for _ in range(episodes):
      state, _ = env.reset()
      term, trunc = False, False
      while not (term or trunc):
        action = self(state, True)
        new_state, reward, term, trunc, _ = env.step(action)
        self.update(state, action, reward, new_state, term)
        state = new_state
      self.eps = max(self.eps_min, self.eps * self.eps_decay)

class MonteCarlo(Model):
  def __init__(self, state_size: int, action_size: int, gamma: float, eps: float = 1.0, eps_min: float = 0.10, eps_decay: float = 0.995):
    self.state_size, self.action_size = state_size, action_size
    self.gamma = gamma
    self.eps, self.eps_min, self.eps_decay = eps, eps_min, eps_decay
    self.reset()

  def reset(self): 
    self.q = np.zeros((self.state_size, self.action_size), dtype="float32")
    self.returns = dict()

  def __call__(self, state: int, initial: bool = False):
    if initial or not self.q[state].any() or np.random.uniform(0, 1) < self.eps: return np.random.choice(self.action_size) # Returns random action
    # if initial and np.random.uniform(0, 1) < self.eps: return np.random.choice(self.action_size)
    else: return self.q[state].argmax() # Returns best action [π(s)] 

  def update(self, state: int, action: int): self.q[state, action] = np.mean(self.returns[(state, action)])
  
  def rollout(self, env: Env, state: int, max_episode_len: int, check_term: bool):
    """ Episode generation continues until term or max_episode_len is reached. """
    act_state = state
    episode = []
    term = False 
    for idx in range(max_episode_len):
        action = self(act_state, not idx)
        new_state, reward, term, trunc, _ = env.step(action)
        episode.append((act_state, action, reward))
        if check_term and term : break
        act_state = new_state
    return episode

  def train(self, env: Env, episodes: int, max_episode_len: int = 20, check_term: bool = True):
    for _ in range(episodes):
        state, _ = env.reset()
        episode = self.rollout(env, state, max_episode_len, check_term) 
        visited = set()
        G = 0
        for i in reversed(episode):
            state, action, reward = i
            G = self.gamma*G + reward
            if (state, action) not in visited:
                visited.add((state, action))
                if (state, action) not in self.returns: self.returns[(state, action)] = []
                self.returns[(state, action)].append(G)
                self.update(state, action)
        self.eps = max(self.eps_min, self.eps * self.eps_decay)

class DynamicProgramming(Model):
  def __init__(self, state_size: int, action_size: int, gamma: float):
    self.state_size, self.action_size = state_size, action_size
    self.gamma = gamma
    self.reset()

  def __call__(self, state: int): return self.q[state].argmax()

  def reset(self): 
    self.q = np.zeros((self.state_size, self.action_size), dtype="float32")
    self.v = np.zeros(self.state_size, dtype="float32")
    self.last_v = np.copy(self.v)
    self.p = None

  def update(self, state: int, action: int):
    self.q[state, action] = sum(prob * (reward + self.gamma * self.last_v[new_state]) for (prob, new_state, reward, _) in self.p[state][action])

  def train(self, env: Env, max_iterations: int = None, threshold: float = 0.001):
    """ Value Iteration algorithm. Iterate until convergence falls below the threshold or the maximum number of iterations is reached. """
    diff = threshold+1e-09; iternum = 0
    self.p = env.unwrapped.P
    while diff > threshold and iternum < max_iterations:
        diff = 0
        self.last_v = np.copy(self.v)
        for state in range(self.state_size):
            for action in range(self.action_size):
                self.update(state, action)
            self.v[state] = np.max(self.q[state])
            diff = max(diff, abs(self.last_v[state] - self.v[state]))
        iternum += 1
        # print(f"diff: {diff} (threshold: {threshold})")
    print(f"\n{iternum} iterations needed.\n" if iternum < max_iterations else f"\nIteration limit exceeded({max_iterations}).\n")
