import gymnasium as gym
from frozenlake.algorithm import Algorithm, Genetic
from frozenlake.const import THREADS
from multiprocessing import Pool

def test(name: str, alg: type[Algorithm], params: dict, train_iter: int, test_iter: int, env_config: dict) -> dict:
  """
  Train an algorithm for `train_iter` cycles and test it for `test_iter` cycles; `env_config` is directly passed to gymnasium.

  Return the statistics data in a dictionary.
  """
  env = gym.make(**env_config)

  data = {}
  a = alg(**params)

  name = a.name if name is None else name
  a.train(env, train_iter)
  steps, success = a.test(env, test_iter)
  data[name] = {
    "train": {
      "steps": list(map(float, a.steps)),
      "actions": {
        "total": list(map(int, a.actions)),
        "history": [list(map(int, r)) for r in a.actions_history],
      },
      "perf": list(map(float, a.performance)),
      "time": list(map(float, a.time))
    },
    "test": {
      "steps": list(map(int, steps)),
      "perf": list(map(int, success))
    }
  }
  if isinstance(a, Genetic): data[name]["policy"] = list(map(int, a.policy))
  else: data[name]["q"] = [[float(c) for c in r] for r in a.q] # type: ignore[attr-defined]

  env.close()

  return data

def schedule_test(config: list, env_config: dict):
  """
  Run all of the algorithms in `config` with the environment config `env_config`.
  Will run in `THREADS` number of CPU cores.

  Return all of the statistics data in a dictionary.
  """
  data = {}
  params = (cfg + (env_config,) for cfg in config)
  with Pool(THREADS) as pool:
    for d in pool.starmap(test, params): data.update(d)
  return data