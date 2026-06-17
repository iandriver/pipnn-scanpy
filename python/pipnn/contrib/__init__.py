"""Optional comparison backends that plug into scanpy the same way PiPNN does.

These wrap third-party ANN libraries as sklearn ``KNeighborsTransformer``s so
they can be compared head-to-head inside ``sc.pp.neighbors`` / the benchmark
notebook. Each import is optional and raises a clear error only when used.
"""

from .glass import GlassTransformer

__all__ = ["GlassTransformer"]
