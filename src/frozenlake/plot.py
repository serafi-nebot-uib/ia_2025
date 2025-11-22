import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from functools import reduce
from operator import getitem
from typing import Any
from frozenlake.algorithm import LearningAlgorithm

def get_item(data: dict[str, Any], *path: str) -> Any: return reduce(getitem, path, data)
def get_data(data: dict[str, Any], *path: str) -> dict: return { k: get_item(data[k], *path) for k in data}

def policy(p: np.ndarray, v: np.ndarray | None, map_desc: np.ndarray, ax: Axes):
  if v is None: v = np.zeros_like(p)
  N = map_desc.shape[0]
  v = v.reshape(N, N)
  p = p.reshape(N, N)

  syms = ["←", "↓", "→", "↑"]
  ax.imshow(v, cmap="Blues")
  for y in range(N):
    for x in range(N):
      if map_desc[y, x] in (b"H", b"G"): continue
      ax.text(x, y, syms[p[y, x]], ha="center", va="center", fontsize=26, color="black", fontweight="bold")

def qtable(data: dict, map_desc: np.ndarray):
  figs = []
  for name in data:
    if "q" in data[name]:
      fig, (axq, axp) = plt.subplots(nrows=1, ncols=2, figsize=(6 * 2, 6))

      q = np.array(data[name]["q"])
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

      policy(p, v, map_desc, axp)

      axq.set_xticks([])
      axq.set_yticks([])
      axp.set_xticks([])
      axp.set_yticks([])
      fig.suptitle(f"{name} Q Table", fontsize=16, fontweight="bold", color="black", ha="center", va="top")
      fig.tight_layout()
    else:
      fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
      p = np.array(data[name]["policy"])
      policy(p, None, map_desc, ax)
      ax.set_xticks([])
      ax.set_yticks([])
      fig.suptitle(f"{name} Policy", fontsize=16, fontweight="bold", color="black", ha="center", va="top")
      fig.tight_layout()
    figs.append(fig)


  return figs

# def policy(desc: np.ndarray, p: np.ndarray, v: np.ndarray | None = None):
#   fig, ax = plt.subplots(figsize=(6, 6))

#   N = int(np.sqrt(len(p)))
#   if v is None: v = np.zeros_like(p)
#   ax.imshow(v.reshape(N, N), cmap="Blues")

#   syms = ["←", "↓", "→", "↑"]
#   for y in range(N):
#     for x in range(N):
#       if desc[y, x] in (b"H", b"G"): continue
#       ax.text(x, y, syms[p[y*N+x]], ha="center", va="center", fontsize=26, color="black", fontweight="bold")

#   ax.set_xticks([])
#   ax.set_yticks([])

#   return fig

def sma(x: list[float] | np.ndarray, m: int) -> np.ndarray: return np.convolve(x, np.ones(m) / m, mode="valid")

def plot_sma(y: list[float] | np.ndarray, label: str, m: int, ax: Axes):
  y = sma(y, m)
  x = np.arange(len(y)) + m
  ax.plot(x, y, label=label)
  ax.set_xlim((0, x.max() + m))

def train_perf(data: dict, m: int = 1):
  data = get_data(data, "train", "perf")
  fig, ax = plt.subplots(figsize=(8, 6))
  for name, values in data.items(): plot_sma(values, name, m, ax)
  ax.set_yticks(np.arange(0, 1.0 + 0.1, 0.1))
  ax.set_ylabel("success rate")
  ax.set_xlabel("iterations")
  agg = f" (SMA {m})" if m > 1 else " "
  fig.suptitle("Train Success Rate" + agg, fontsize=16, fontweight="bold", color="black", ha="center", va="top")
  fig.legend(loc="upper right")
  fig.tight_layout()
  return fig

def train_time(data: dict, m: int = 1):
  data = get_data(data, "train", "time")
  fig, ax = plt.subplots(figsize=(8, 6))
  for name, values in data.items(): plot_sma(values, name, m, ax)
  ax.set_ylabel("time (s)")
  ax.set_xlabel("iterations")
  agg = f" (SMA {m})" if m > 1 else " "
  fig.suptitle("Train Iteration Time" + agg, fontsize=16, fontweight="bold", color="black", ha="center", va="top")
  fig.legend(loc="upper right")
  return fig

def train_steps(data: dict, m: int = 1):
  data = get_data(data,"train", "steps")
  fig, ax = plt.subplots(figsize=(8, 6))
  for name, values in data.items(): plot_sma(values, name, m, ax)
  ax.set_ylabel("steps")
  ax.set_xlabel("iterations")
  agg = f" (SMA {m})" if m > 1 else " "
  fig.suptitle("Train Episode Steps" + agg, fontsize=16, fontweight="bold", color="black", ha="center", va="top")
  fig.legend(loc="upper right")
  return fig

def train_actions(data: dict):
  data = get_data(data, "train", "actions", "total")
  fig, ax = plt.subplots(figsize=(8, 6))
  width = 0.45
  
  # TODO: do not hardcode (get from env?)
  actions = ["left", "down", "right", "up"]
  x = np.arange(len(actions))
  for i, (label, values) in enumerate(data.items()):
    ax.bar(x + i * len(actions), values, width=width, label=label)

  ax.set_xticks(np.arange(len(actions) * len(data)), actions * len(data))
  ax.set_ylabel("number of actions")
  fig.suptitle("Train Chosen Actions", fontsize=16, fontweight="bold", color="black", ha="center", va="top")
  fig.legend()
  fig.tight_layout()

def train_actions_history(data: dict, m: int):
  data = get_data(data, "train", "actions", "history")
  actions = ["left", "down", "right", "up"]
  fig, ax = plt.subplots(figsize=(8, 6))
  for i, (label, values) in enumerate(data.items()):
    v = np.array(values)
    v = v.reshape(v.shape[0] // m, -1, v.shape[1]).mean(axis=1)
    ax.imshow(v)
  # ax.set_xticklabels(actions)
  ax.set_ylabel("number of iterations")
  
  # # TODO: do not hardcode (get from env?)
  # actions = ["left", "down", "right", "up"]
  # x = np.arange(len(actions))
  # for i, (label, values) in enumerate(data.items()):
  #   ax.bar(x + i * len(actions), values, width=width, label=label)

  # ax.set_xticks(np.arange(len(actions) * len(data)), actions * len(actions))
  # ax.set_ylabel("number of actions")
  # fig.suptitle("Train Chosen Actions", fontsize=16, fontweight="bold", color="black", ha="center", va="top")
  # fig.legend()
  # fig.tight_layout()

def test_steps(data: dict):
  data = get_data(data, "test", "steps")
  fig, ax = plt.subplots(figsize=(8, 6))
  labels, values = zip(*data.items())
  ax.boxplot(values, whis=(0, 100), vert=True)
  ax.set_xticklabels(labels, rotation=90)
  ax.set_ylabel("steps")
  fig.suptitle("Test Steps in Episode", fontsize=16, fontweight="bold", color="black", ha="center", va="top")
  fig.tight_layout()
  return fig