import numpy as np
from tinyml_compiler.benchmark import benchmark, estimate_memory_bytes
from tinyml_compiler.builder import mlp_graph
from tinyml_compiler.passes import optimize


def main() -> None:
    g = mlp_graph(batch=32, in_dim=64, hidden=128, out_dim=32)
    x = np.random.default_rng(0).standard_normal((32, 64)).astype(np.float32)

    baseline_ms = benchmark(g, {"x": x}, runs=300)
    baseline_mem = estimate_memory_bytes(g)

    opt = optimize(g)
    opt_ms = benchmark(opt, {"x": x}, runs=300)
    opt_mem = estimate_memory_bytes(opt)

    print("| Variant | Latency (ms) | Memory (KB) |")
    print("|---|---:|---:|")
    print(f"| Baseline | {baseline_ms:.4f} | {baseline_mem / 1024:.2f} |")
    print(f"| Optimized | {opt_ms:.4f} | {opt_mem / 1024:.2f} |")


if __name__ == "__main__":
    main()
