from __future__ import annotations

import time
import numpy as np
from .runtime import execute_graph


def estimate_memory_bytes(graph) -> int:
    total = 0
    for t in graph.tensors.values():
        if len(t.shape) == 0:
            continue
        item = np.dtype(t.dtype if t.dtype != "int8" else np.int8).itemsize
        n = int(np.prod(t.shape))
        total += n * item
    return total


def benchmark(graph, feeds, runs: int = 200) -> float:
    for _ in range(20):
        execute_graph(graph, feeds)
    start = time.perf_counter()
    for _ in range(runs):
        execute_graph(graph, feeds)
    end = time.perf_counter()
    return (end - start) * 1000 / runs
