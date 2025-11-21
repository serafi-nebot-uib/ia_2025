from tinygrad import Tensor, nn
from collections import deque

class DQN:
  def __init__(self, state_size: int, action_size: int, hidden_size: int):
    self.state_size, self.action_size, self.hidden_size = state_size, action_size, hidden_size
    self.l1 = nn.Linear(state_size, hidden_size)
    self.l2 = nn.Linear(hidden_size, hidden_size)
    self.l3 = nn.Linear(hidden_size, action_size)

  def __call__(self, state: Tensor) -> Tensor:
    x = state.one_hot(num_classes=self.state_size)
    x = self.l1(x)
    x = self.l2(x)
    x = self.l3(x)
    return x.argmax()

class Buffer:
  def __init__(self, maxlen: int): self.maxlen, self.buffer = maxlen, deque(maxlen=maxlen)

  def push(self, state, action, reward, next_state, done): self.buffer.append((state, action, reward, next_state, done))

  def sample(self, batch_size):
    Tensor.randint(low=0, high=batch_size)


m = DQN(16, 4, 128)
x = Tensor(6)
print(m(x).numpy())