import numpy as np

from tinyml_compiler.builder import mlp_graph
from tinyml_compiler.ir import Graph, Node, Tensor
from tinyml_compiler.passes import (
    cleanup_consecutive_transposes,
    constant_folding,
    dead_node_elimination,
    fuse_linear_relu,
    optimize,
)
from tinyml_compiler.runtime import execute_graph


def test_fuse_linear_relu():
    g = mlp_graph()
    fused = fuse_linear_relu(g)
    assert any(n.op_type == "FusedLinearReLU" for n in fused.nodes)


def test_constant_folding_add():
    g = Graph("const")
    g.outputs = ["z"]
    g.add_tensor(Tensor("a", (2,), data=np.array([1.0, 2.0], dtype=np.float32)))
    g.add_tensor(Tensor("b", (2,), data=np.array([2.0, 3.0], dtype=np.float32)))
    g.add_node(Node("add", "Add", ["a", "b"], ["z"]))
    folded = constant_folding(g)
    assert len(folded.nodes) == 0
    np.testing.assert_allclose(folded.tensors["z"].data, np.array([3.0, 5.0], dtype=np.float32))


def test_dead_node_elimination():
    g = Graph("dead")
    g.outputs = ["keep"]
    g.add_tensor(Tensor("x", (2,), data=np.array([1.0, 2.0], dtype=np.float32)))
    g.add_tensor(Tensor("y", (2,), data=np.array([3.0, 4.0], dtype=np.float32)))
    g.add_node(Node("dead_add", "Add", ["x", "y"], ["dead"]))
    g.add_node(Node("live_relu", "ReLU", ["x"], ["keep"]))
    pruned = dead_node_elimination(g)
    assert [n.name for n in pruned.nodes] == ["live_relu"]


def test_transpose_cleanup():
    g = Graph("tr")
    g.outputs = ["y"]
    g.add_node(Node("t1", "Transpose", ["x"], ["m"], attrs={"axes": (1, 0)}))
    g.add_node(Node("t2", "Transpose", ["m"], ["y"], attrs={"axes": (1, 0)}))
    out = cleanup_consecutive_transposes(g)
    assert len(out.nodes) == 0


def test_optimized_matches_unoptimized_tolerance():
    g = mlp_graph(batch=8, in_dim=8, hidden=32, out_dim=8)
    x = np.random.default_rng(123).standard_normal((8, 8)).astype(np.float32)
    ref = execute_graph(g, {"x": x})["prob"]
    opt = optimize(g)
    got = execute_graph(opt, {"x": x})["prob"]
    np.testing.assert_allclose(ref, got, atol=2e-2, rtol=2e-2)
