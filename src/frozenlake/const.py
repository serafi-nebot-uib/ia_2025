from typing import Any
import os

def getenv(key: str, default: Any = 0): return type(default)(os.getenv(key, default))

DEBUG = getenv("DEBUG", 0)
STATS = getenv("STATS", 0)
SAMPLE = getenv("SAMPLE", 1000)
THREADS_MAX = os.cpu_count()
THREADS = getenv("THREADS", 1)
if THREADS < 1 or THREADS > THREADS_MAX: THREADS = THREADS_MAX