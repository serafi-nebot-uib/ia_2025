import json
from pathlib import Path
import gymnasium as gym
from frozenlake.algorithm import Algorithm, LearningAlgorithm

def test(config: list[tuple[str | None, type[Algorithm], dict, int, int]]):
  env = gym.make("FrozenLake-v1", is_slippery=True, render_mode=None)

  data = {}
  for name, alg, params, train_iter, test_iter in config:
    a = alg(**params)

    name = a.name if name is None else name
    a.train(env, train_iter)
    steps, success = a.test(env, test_iter)
    data[name] = {
      "train": {
        "steps": list(map(float, a.steps)),
        "actions": list(map(int, a.actions)),
        "perf": list(map(float, a.performance)),
        "time": list(map(float, a.time))
      },
      "test": {
        "steps": list(map(int, steps)),
        "perf": list(map(int, success))
      }
    }
    if isinstance(a, LearningAlgorithm): data[name]["q"] = [[float(c) for c in r] for r in a.q]

  env.close()

  return data

def run_test(name, config):
  p = Path("data/" + name + ".json")
  if not p.parent.exists(): p.parent.mkdir()
  with p.open("w") as f:
    data = test(config)
    json.dump(data, f)