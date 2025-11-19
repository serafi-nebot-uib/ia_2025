import numpy as np
import matplotlib.pyplot as plt

def qtable(desc: np.ndarray, q: np.ndarray):
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
      if desc[y, x] in (b"H", b"G"): continue
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
      if desc[y, x] in (b"H", b"G"): continue
      axp.text(x, y, syms[p[y, x]], ha="center", va="center", fontsize=26, color="black", fontweight="bold")

  axq.set_xticks([])
  axq.set_yticks([])
  axp.set_xticks([])
  axp.set_yticks([])

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