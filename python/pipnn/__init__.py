"""PiPNN: fast graph-based nearest-neighbor indexing for single-cell / scanpy data."""

from .transformer import PiPNNTransformer
from .index import self_knn_graph

try:
    from ._pipnn import __version__
except Exception:  # pragma: no cover - extension not built yet
    __version__ = "0.0.0"

__all__ = ["PiPNNTransformer", "self_knn_graph", "__version__"]
