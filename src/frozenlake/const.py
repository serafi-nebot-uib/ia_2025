from typing import Any
import os

def getenv(key: str, default: Any = 0): return type(default)(os.getenv(key, default))

DEBUG = getenv("DEBUG", 0)
STATS = getenv("STATS", 0)
SAMPLE = getenv("SAMPLE", 1000)