import os
from typing import Any

# get the value of an environment variable and cast it to the default type
def getenv(key: str, default: Any = 0): return type(default)(os.getenv(key, default))

DEBUG = getenv("DEBUG", 0) # show debug messages with increasing level of verbosity (0 -> disabled)
STATS = getenv("STATS", 0) # track algorithm statistics
SAMPLE = getenv("SAMPLE", 1000) # show debug messages every SAMPLE iteration when training
RAND_TIE_BREAK = bool(getenv("RAND_TIE_BREAK", 1)) # enable/disable random tie-break for LearningAlgorithm

# maximum number of cores to run tests on (THREADS < 1 -> all cpu cores available)
__THREADS_MAX = os.cpu_count()
THREADS = getenv("THREADS", 1)
if THREADS < 1 or THREADS > __THREADS_MAX: THREADS = __THREADS_MAX

if DEBUG > 1:
  print("options: ")
  print(f"         DEBUG: {DEBUG}")
  print(f"         STATS: {STATS}")
  print(f"        SAMPLE: {SAMPLE}")
  print(f"       THREADS: {THREADS}")
  print(f"RAND_TIE_BREAK: {RAND_TIE_BREAK}")