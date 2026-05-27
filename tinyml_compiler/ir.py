from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np


@dataclass
class Tensor:
    name: str
    shape: tuple[int, ...]
    dtype: str = "float32"
    data: Optional[np.ndarray] = None

    @property
    def is_constant(self) -> bool:
        return self.data is not None


@dataclass
class Node:
    name: str
    op_type: str
    inputs: List[str]
    outputs: List[str]
    attrs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Graph:
    name: str
    tensors: Dict[str, Tensor] = field(default_factory=dict)
    nodes: List[Node] = field(default_factory=list)
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)

    def add_tensor(self, tensor: Tensor) -> None:
        self.tensors[tensor.name] = tensor

    def add_node(self, node: Node) -> None:
        self.nodes.append(node)

    def copy(self) -> "Graph":
        new = Graph(name=self.name)
        new.inputs = list(self.inputs)
        new.outputs = list(self.outputs)
        new.tensors = {
            k: Tensor(
                name=v.name,
                shape=v.shape,
                dtype=v.dtype,
                data=None if v.data is None else v.data.copy(),
            )
            for k, v in self.tensors.items()
        }
        new.nodes = [
            Node(
                name=n.name,
                op_type=n.op_type,
                inputs=list(n.inputs),
                outputs=list(n.outputs),
                attrs=dict(n.attrs),
            )
            for n in self.nodes
        ]
        return new
