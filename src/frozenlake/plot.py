import numpy as np
import matplotlib.pyplot as plt

def qtable(q: np.ndarray, nrows: int, ncols: int):
  fig, ax = plt.subplots(figsize=(8, 8))

  values = q.max(axis=-1).reshape(nrows, ncols)
  ax.imshow(values, cmap="Blues")

  # apply softmax to get arrow lengths proportionate to each action's value
  shifted_q = q - q.max(axis=-1, keepdims=True) # shift for numerical stability
  exp_q = np.exp(shifted_q)
  anorm = (exp_q / exp_q.sum(axis=-1, keepdims=True)).reshape(nrows, ncols, -1, 1)
  #                 left     down    right   up
  adir = np.array([[-1, 0], [0, 1], [1, 0], [0, -1]])
  arrow = adir * anorm

  for y in range(arrow.shape[0]):
    for x in range(arrow.shape[1]):
      if values[y, x] == 0: continue
      for a in range(arrow.shape[2]):
        dx, dy = arrow[y, x, a]
        ax.arrow(x, y, dx, dy,
                 head_width=0.08, head_length=0.08,
                 fc="white", ec="black", alpha=0.8,
                 length_includes_head=True, clip_on=True)

  ax.set_xticks([])
  ax.set_yticks([])
  plt.tight_layout()
  plt.show()