"""tinyml-compiler-lab: educational ML compiler and runtime."""

from .ir import Tensor, Node, Graph
from .runtime import execute_graph

__all__ = ["Tensor", "Node", "Graph", "execute_graph"]
