from . import Model
import numpy as np
from gymnasium import Env

# TODO: monte carlo doesn't match Model's structure

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

  def update(self, state: int, action: int):
    self.q[state, action] = np.mean(self.returns[(state, action)])
  
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