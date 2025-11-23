from typing import Any
import os

def getenv(key: str, default: Any = 0): return type(default)(os.getenv(key, default))

DEBUG = getenv("DEBUG", 0)
STATS = getenv("STATS", 0)
SAMPLE = getenv("SAMPLE", 1000)
RAND_TIE_BREAK = getenv("RAND_TIE_BREAK", 1)
THREADS_MAX = os.cpu_count()
THREADS = getenv("THREADS", 1)
if THREADS < 1 or THREADS > THREADS_MAX: THREADS = THREADS_MAX

if DEBUG > 1:
  print("options: ")
  print(f"         DEBUG: {DEBUG}")
  print(f"         STATS: {STATS}")
  print(f"        SAMPLE: {SAMPLE}")
  print(f"       THREADS: {THREADS}")
  print(f"RAND_TIE_BREAK: {RAND_TIE_BREAK}")