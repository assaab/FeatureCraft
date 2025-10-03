"""Distributed executors for feature plan execution."""

from .pandas_executor import PandasExecutor
from .spark_executor import SparkExecutor

__all__ = [
    "PandasExecutor",
    "SparkExecutor",
]

# Conditionally import Ray executor (requires ray package)
try:
    from .ray_executor import RayExecutor
    __all__.append("RayExecutor")
except (ImportError, NameError):
    RayExecutor = None

