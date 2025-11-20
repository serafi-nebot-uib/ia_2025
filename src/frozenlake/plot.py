import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

def qtable(name: str, q: np.ndarray, map_desc: np.ndarray):
  fig, (axq, axp) = plt.subplots(nrows=1, ncols=2, figsize=(6 * 2, 6))

  N = int(np.sqrt(q.shape[0]))
  v = q.max(axis=-1).reshape(N, N)
  p = q.argmax(axis=-1).reshape(N, N)

  axq.imshow(v, cmap="Blues")

  # apply softmax to get arrow lengths proportionate to each action's value
  shifted_q = q - q.max(axis=-1, keepdims=True) # shift for numerical stability
  exp_q = np.exp(shifted_q)

  anorm = (exp_q / exp_q.sum(axis=-1, keepdims=True)).reshape(N, N, -1, 1)
  #                 left     down    right   up
  adir = np.array([[-1, 0], [0, 1], [1, 0], [0, -1]])
  arrow = adir * anorm

  for y in range(arrow.shape[0]):
    for x in range(arrow.shape[1]):
      if map_desc[y, x] in (b"H", b"G"): continue
      for a in range(arrow.shape[2]):
        dx, dy = arrow[y, x, a]
        axq.arrow(x, y, dx, dy,
                 head_width=0.08, head_length=0.08,
                 fc="white", ec="black", alpha=0.8,
                 length_includes_head=True, clip_on=True)

  syms = ["←", "↓", "→", "↑"]
  axp.imshow(v, cmap="Blues")
  for y in range(N):
    for x in range(N):
      if map_desc[y, x] in (b"H", b"G"): continue
      axp.text(x, y, syms[p[y, x]], ha="center", va="center", fontsize=26, color="black", fontweight="bold")

  axq.set_xticks([])
  axq.set_yticks([])
  axp.set_xticks([])
  axp.set_yticks([])
  fig.suptitle(f"{name} Q Table", fontsize=16, fontweight="bold", color="black", ha="center", va="top")
  fig.tight_layout()

  return fig

def policy(desc: np.ndarray, p: np.ndarray, v: np.ndarray | None = None):
  fig, ax = plt.subplots(figsize=(6, 6))

  N = int(np.sqrt(len(p)))
  if v is None: v = np.zeros_like(p)
  ax.imshow(v.reshape(N, N), cmap="Blues")

  syms = ["←", "↓", "→", "↑"]
  for y in range(N):
    for x in range(N):
      if desc[y, x] in (b"H", b"G"): continue
      ax.text(x, y, syms[p[y*N+x]], ha="center", va="center", fontsize=26, color="black", fontweight="bold")

  ax.set_xticks([])
  ax.set_yticks([])

  return fig

def sma(data: dict[str, list[float] | np.ndarray], m: int) -> dict[str, list[float] | np.ndarray]:
  return { name: np.convolve(value, np.ones(m) / m, mode="valid") for name, value in data.items() }

def plot_data(data: dict[str, list[float] | np.ndarray], m: int, ax: Axes):
  xmax = 0
  for name, y in data.items():
    x = np.arange(len(y)) + m
    ax.plot(x, y, label=name)
    xmax = max(xmax, x.max())
  ax.set_xlim((0, xmax + m))

def success_rate(data: dict[str, list[float] | np.ndarray], m: int = 1):
  fig, ax = plt.subplots(figsize=(8, 6))
  if m > 1: data = sma(data, m)
  plot_data(data, m, ax)
  ax.set_yticks(np.arange(0, 1.0 + 0.1, 0.1))
  ax.set_ylabel("success rate")
  ax.set_xlabel("iterations")
  agg = f" (SMA {m})" if m > 1 else " "
  fig.suptitle("Train Success Rate" + agg, fontsize=16, fontweight="bold", color="black", ha="center", va="top")
  fig.legend(loc="upper right")
  fig.tight_layout()
  return fig

def train_time(data: dict[str, list[float] | np.ndarray], m: int = 1):
  fig, ax = plt.subplots(figsize=(8, 6))
  if m > 1: data = sma(data, m)
  plot_data(data, m, ax)
  ax.set_ylabel("time (s)")
  ax.set_xlabel("iterations")
  agg = f" (SMA {m})" if m > 1 else " "
  fig.suptitle("Train Iteration Time" + agg, fontsize=16, fontweight="bold", color="black", ha="center", va="top")
  fig.legend(loc="upper right")
  return fig