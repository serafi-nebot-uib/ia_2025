import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt

# -----------------------------
# Deep Q-Network
# -----------------------------
class DQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# -----------------------------
# Replay Buffer
# -----------------------------
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (np.stack(state), np.array(action), np.array(reward),
                np.stack(next_state), np.array(done))

    def __len__(self):
        return len(self.buffer)

# -----------------------------
# Hyperparameters
# -----------------------------
env = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=True)  # 8x8 is harder
# env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False)  # deterministic

state_size  = env.observation_space.n      # 64 for 8x8, 16 for 4x4
action_size = env.action_space.n           # 4 actions

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

policy_net = DQN(state_size, action_size).to(device)
target_net = DQN(state_size, action_size).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
buffer = ReplayBuffer(capacity=10000)

# DQN hyperparameters
batch_size = 128
gamma      = 0.99
epsilon    = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
target_update = 10
num_episodes = 2000
max_steps = 200

# One-hot encoding for discrete states
def one_hot(state):
    vec = np.zeros(state_size)
    vec[state] = 1
    return vec

# -----------------------------
# Training Loop
# -----------------------------
rewards = []

for episode in range(num_episodes):
    state, _ = env.reset()
    state = one_hot(state)
    total_reward = 0

    for t in range(max_steps):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

        # Epsilon-greedy action selection
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            q_values = policy_net(state_tensor)
            action = q_values.argmax().item()

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        next_state = one_hot(next_state)

        # Store transition
        buffer.push(state, action, reward, next_state, done)

        state = next_state
        total_reward += reward

        # Update network
        if len(buffer) >= batch_size:
            states, actions, rewards_batch, next_states, dones = buffer.sample(batch_size)

            states      = torch.FloatTensor(states).to(device)
            actions     = torch.LongTensor(actions).unsqueeze(1).to(device)
            rewards_b   = torch.FloatTensor(rewards_batch).to(device)
            next_states = torch.FloatTensor(next_states).to(device)
            dones       = torch.FloatTensor(dones).to(device)

            current_q = policy_net(states).gather(1, actions).squeeze(1)
            next_q    = target_net(next_states).max(1)[0]
            target_q  = rewards_b + gamma * next_q * (1 - dones)

            loss = F.mse_loss(current_q, target_q.detach())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if done:
            break

    # Decay epsilon
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    # Update target network
    if episode % target_update == 0:
        target_net.load_state_dict(policy_net.state_dict())

    rewards.append(total_reward)

    if (episode+1) % 100 == 0:
        avg_reward = np.mean(rewards[-100:])
        print(f"Episode {episode+1}/{num_episodes} | Avg Reward (last 100): {avg_reward:.3f} | Epsilon: {epsilon:.3f}")

print("Training finished!")

# -----------------------------
# Plot results
# -----------------------------
plt.plot(rewards, label="Reward per episode")
plt.plot(np.convolve(rewards, np.ones(100)/100, mode='valid'), label="100-episode avg")
plt.axhline(y=1.0, color='r', linestyle='--', label="Goal reached")
plt.legend()
plt.title("DQN on FrozenLake-v1 (8x8, slippery)")
plt.show()

# -----------------------------
# Test the trained agent
# -----------------------------
env = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=True, render_mode="human")
state, _ = env.reset()
state = one_hot(state)

for _ in range(500):
    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
    action = policy_net(state_tensor).argmax().item()
    next_state, reward, terminated, truncated, _ = env.step(action)
    state = one_hot(next_state)
    if terminated or truncated:
        break

env.close()