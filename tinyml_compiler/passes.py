from __future__ import annotations

from typing import Set
import numpy as np

from .ir import Graph, Node, Tensor
from .runtime import execute_graph


def constant_folding(graph: Graph) -> Graph:
    g = graph.copy()
    changed = True
    while changed:
        changed = False
        new_nodes = []
        for node in g.nodes:
            if all(g.tensors[i].is_constant for i in node.inputs if i in g.tensors):
                feeds = {n: g.tensors[n].data for n in node.inputs}
                temp = Graph(name="tmp", tensors=g.tensors, nodes=[node], outputs=node.outputs)
                out = execute_graph(temp, feeds)[node.outputs[0]]
                g.tensors[node.outputs[0]] = Tensor(name=node.outputs[0], shape=out.shape, data=out)
                changed = True
            else:
                new_nodes.append(node)
        g.nodes = new_nodes
    return g


def dead_node_elimination(graph: Graph) -> Graph:
    g = graph.copy()
    needed: Set[str] = set(g.outputs)
    kept = []
    for node in reversed(g.nodes):
        if any(o in needed for o in node.outputs):
            kept.append(node)
            needed.update(node.inputs)
    g.nodes = list(reversed(kept))
    return g


def fuse_linear_relu(graph: Graph) -> Graph:
    g = graph.copy()
    out_nodes = []
    i = 0
    while i < len(g.nodes):
        node = g.nodes[i]
        if (
            i + 1 < len(g.nodes)
            and node.op_type == "Linear"
            and g.nodes[i + 1].op_type == "ReLU"
            and g.nodes[i + 1].inputs == [node.outputs[0]]
        ):
            relu = g.nodes[i + 1]
            out_nodes.append(
                Node(
                    name=f"{node.name}_relu_fused",
                    op_type="FusedLinearReLU",
                    inputs=list(node.inputs),
                    outputs=list(relu.outputs),
                    attrs=dict(node.attrs),
                )
            )
            i += 2
        else:
            out_nodes.append(node)
            i += 1
    g.nodes = out_nodes
    return g


def cleanup_consecutive_transposes(graph: Graph) -> Graph:
    g = graph.copy()
    out_nodes = []
    i = 0
    while i < len(g.nodes):
        if i + 1 < len(g.nodes):
            a, b = g.nodes[i], g.nodes[i + 1]
            if a.op_type == b.op_type == "Transpose" and b.inputs == [a.outputs[0]]:
                ax1 = a.attrs.get("axes")
                ax2 = b.attrs.get("axes")
                if ax1 is not None and ax2 is not None:
                    composed = tuple(ax1[j] for j in ax2)
                    if composed == tuple(range(len(composed))):
                        i += 2
                        continue
        out_nodes.append(g.nodes[i])
        i += 1
    g.nodes = out_nodes
    return g


def simulate_int8_linear_weights(graph: Graph) -> Graph:
    g = graph.copy()
    for node in g.nodes:
        if node.op_type in {"Linear", "FusedLinearReLU"}:
            w_name = node.inputs[1]
            w = g.tensors[w_name].data
            if w is None:
                continue
            max_abs = float(np.max(np.abs(w)))
            scale = max_abs / 127.0 if max_abs > 0 else 1.0
            q = np.clip(np.round(w / scale), -127, 127).astype(np.int8)
            g.tensors[w_name] = Tensor(name=w_name, shape=w.shape, dtype="int8", data=q)
            node.attrs["quantized_weight"] = True
            node.attrs["weight_scale"] = scale
    return g


def optimize(graph: Graph, *, quantize: bool = False) -> Graph:
    """Run graph optimization passes.

    By default, optimize() only runs semantics-preserving passes.
    Quantization changes numerical values, so it is opt-in.
    """
    g = constant_folding(graph)
    g = fuse_linear_relu(g)
    g = cleanup_consecutive_transposes(g)
    g = dead_node_elimination(g)

    if quantize:
        g = simulate_int8_linear_weights(g)

    return g
