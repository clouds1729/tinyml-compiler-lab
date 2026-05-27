from __future__ import annotations

from typing import Dict
import numpy as np

from .ir import Graph
from . import ops


def _dequantize_weight(qweight: np.ndarray, scale: float) -> np.ndarray:
    return (qweight.astype(np.float32) * scale).astype(np.float32)


def execute_graph(graph: Graph, feeds: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    values: Dict[str, np.ndarray] = {}
    for name, tensor in graph.tensors.items():
        if tensor.data is not None:
            values[name] = tensor.data
    values.update(feeds)

    for node in graph.nodes:
        ins = [values[i] for i in node.inputs]
        if node.op_type == "MatMul":
            out = ops.matmul(ins[0], ins[1])
        elif node.op_type == "Add":
            out = ops.add(ins[0], ins[1])
        elif node.op_type == "ReLU":
            out = ops.relu(ins[0])
        elif node.op_type == "Linear":
            x, w = ins[0], ins[1]
            if node.attrs.get("quantized_weight", False):
                scale = float(node.attrs["weight_scale"])
                w = _dequantize_weight(w, scale)
            out = ops.matmul(x, w.T)
            if len(ins) > 2:
                out = ops.add(out, ins[2])
        elif node.op_type == "FusedLinearReLU":
            x, w = ins[0], ins[1]
            if node.attrs.get("quantized_weight", False):
                scale = float(node.attrs["weight_scale"])
                w = _dequantize_weight(w, scale)
            out = ops.matmul(x, w.T)
            if len(ins) > 2:
                out = ops.add(out, ins[2])
            out = ops.relu(out)
        elif node.op_type == "Softmax":
            out = ops.softmax(ins[0], axis=node.attrs.get("axis", -1))
        elif node.op_type == "LayerNorm":
            out = ops.layernorm(ins[0], eps=node.attrs.get("eps", 1e-5))
        elif node.op_type == "Transpose":
            out = np.transpose(ins[0], axes=node.attrs.get("axes"))
        else:
            raise ValueError(f"Unsupported op_type: {node.op_type}")

        values[node.outputs[0]] = out

    return {name: values[name] for name in graph.outputs}
