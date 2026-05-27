from __future__ import annotations

import statistics
import time
from dataclasses import dataclass

import numpy as np

from tinyml_compiler.benchmark import estimate_memory_bytes
from tinyml_compiler.builder import mlp_graph
from tinyml_compiler.passes import optimize
from tinyml_compiler.runtime import execute_graph


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    batch: int
    in_dim: int
    hidden: int
    out_dim: int
    runs: int


CASES = [
    BenchmarkCase("tiny", batch=32, in_dim=64, hidden=128, out_dim=32, runs=300),
    BenchmarkCase("small", batch=64, in_dim=256, hidden=512, out_dim=128, runs=100),
    BenchmarkCase("medium", batch=128, in_dim=512, hidden=2048, out_dim=512, runs=30),
    BenchmarkCase("large", batch=128, in_dim=1024, hidden=4096, out_dim=1024, runs=10),
]


def time_run(graph, x: np.ndarray, runs: int) -> float:
    # Warmup: let NumPy allocate/cache whatever it needs.
    for _ in range(3):
        execute_graph(graph, {"x": x})

    times_ms = []
    for _ in range(runs):
        start = time.perf_counter()
        execute_graph(graph, {"x": x})
        end = time.perf_counter()
        times_ms.append((end - start) * 1000.0)

    return statistics.median(times_ms)


def max_abs_error(graph, opt_graph, x: np.ndarray) -> float:
    ref = execute_graph(graph, {"x": x})["prob"]
    got = execute_graph(opt_graph, {"x": x})["prob"]
    return float(np.max(np.abs(ref - got)))


def main() -> None:
    rng = np.random.default_rng(123)

    print(
        "| Case | Shape | Nodes Base -> Opt | Baseline ms | Optimized ms | "
        "Speedup | Memory Base KB | Memory Opt KB | Max Abs Error |"
    )
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|")

    for case in CASES:
        graph = mlp_graph(
            batch=case.batch,
            in_dim=case.in_dim,
            hidden=case.hidden,
            out_dim=case.out_dim,
        )
        opt_graph = optimize(graph)

        x = rng.standard_normal((case.batch, case.in_dim)).astype(np.float32)

        baseline_ms = time_run(graph, x, case.runs)
        optimized_ms = time_run(opt_graph, x, case.runs)
        speedup = baseline_ms / optimized_ms if optimized_ms > 0 else float("inf")

        mem_base_kb = estimate_memory_bytes(graph) / 1024.0
        mem_opt_kb = estimate_memory_bytes(opt_graph) / 1024.0

        err = max_abs_error(graph, opt_graph, x)

        shape = f"{case.batch}x{case.in_dim}->{case.hidden}->{case.out_dim}"
        nodes = f"{len(graph.nodes)} -> {len(opt_graph.nodes)}"

        print(
            f"| {case.name} | {shape} | {nodes} | "
            f"{baseline_ms:.4f} | {optimized_ms:.4f} | {speedup:.2f}x | "
            f"{mem_base_kb:.2f} | {mem_opt_kb:.2f} | {err:.2e} |"
        )


if __name__ == "__main__":
    main()