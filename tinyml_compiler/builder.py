from __future__ import annotations

import numpy as np
from .ir import Graph, Node, Tensor


def mlp_graph(batch: int = 4, in_dim: int = 8, hidden: int = 16, out_dim: int = 4) -> Graph:
    g = Graph(name="mlp")
    g.inputs = ["x"]
    g.outputs = ["prob"]

    rng = np.random.default_rng(0)
    g.add_tensor(Tensor("w1", (hidden, in_dim), data=rng.standard_normal((hidden, in_dim)).astype(np.float32)))
    g.add_tensor(Tensor("b1", (hidden,), data=rng.standard_normal((hidden,)).astype(np.float32)))
    g.add_tensor(Tensor("w2", (out_dim, hidden), data=rng.standard_normal((out_dim, hidden)).astype(np.float32)))
    g.add_tensor(Tensor("b2", (out_dim,), data=rng.standard_normal((out_dim,)).astype(np.float32)))

    g.add_node(Node("linear1", "Linear", ["x", "w1", "b1"], ["h1"]))
    g.add_node(Node("relu1", "ReLU", ["h1"], ["h1_relu"]))
    g.add_node(Node("linear2", "Linear", ["h1_relu", "w2", "b2"], ["logits"]))
    g.add_node(Node("softmax", "Softmax", ["logits"], ["prob"], attrs={"axis": -1}))

    g.add_tensor(Tensor("x", (batch, in_dim), dtype="float32"))
    g.add_tensor(Tensor("h1", (batch, hidden), dtype="float32"))
    g.add_tensor(Tensor("h1_relu", (batch, hidden), dtype="float32"))
    g.add_tensor(Tensor("logits", (batch, out_dim), dtype="float32"))
    g.add_tensor(Tensor("prob", (batch, out_dim), dtype="float32"))
    return g
