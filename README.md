# tinyml-compiler-lab

A small educational ML compiler/runtime for understanding how neural network computation graphs are represented, optimized, and executed.

The goal is not to compete with TVM, XLA, ONNX Runtime, TensorRT, or other production inference stacks. The goal is to make the core ideas visible in a compact codebase: graph IRs, compiler passes, numerical correctness tests, approximate quantization, and simple latency/memory benchmarking.

## What this project demonstrates

- Minimal graph IR: `Tensor`, `Node`, and `Graph`
- Python graph construction path
- Optimization passes:
  - constant folding
  - dead-node elimination
  - Linear + ReLU fusion
  - consecutive transpose cleanup
- Opt-in int8 weight quantization simulation for linear layers
- NumPy runtime for executing graphs
- Correctness tests comparing optimized and unoptimized graph outputs
- Benchmarking harness for latency and memory estimates

## Why this matters

Modern ML inference stacks depend on compiler-style transformations. A model is not just "run" directly; its computation graph is lowered, simplified, fused, quantized, scheduled, and eventually mapped onto CPU, GPU, DSP, NPU, or custom accelerator backends.

This project explores a tiny version of that pipeline.

## Quick start

```bash
python3.11 -m venv .venv
source .venv/bin/activate

pip install -e ".[dev]"
pytest -q
python benchmarks/run_benchmark.py
````

Expected test result:

```text
5 passed
```

Example benchmark output:

```text
| Variant   | Latency (ms) | Memory (KB) |
|-----------|-------------:|------------:|
| Baseline  |       0.0650 |       96.62 |
| Optimized |       0.0642 |       96.62 |
```

The benchmark graph is intentionally small, so speedups are modest. The point is to demonstrate the compiler pipeline and measurement harness rather than claim production performance.

## Compiler pipeline

1. **Frontend / graph construction**

   Build a neural-network-style graph in Python.

2. **Optimization**

   Run graph passes such as constant folding, dead-node elimination, operator fusion, and transpose cleanup.

3. **Runtime execution**

   Execute the graph using NumPy reference kernels.

4. **Testing and benchmarking**

   Compare optimized and unoptimized outputs for numerical correctness, then measure latency and estimated memory usage.

## Semantics-preserving vs approximate passes

By default, `optimize(graph)` only runs semantics-preserving passes. These are transformations that should preserve the graph output up to normal floating-point tolerance.

Approximate transforms such as int8 weight quantization are opt-in:

```python
opt = optimize(graph, quantize=True)
```

Quantization can change numerical outputs, so it is tested and reasoned about separately from exact compiler optimizations.

## Qualcomm-relevant systems perspective

This project is motivated by edge, mobile, and accelerator-aware ML deployment.

Key systems ideas:

* **Latency:** operator fusion can reduce runtime overhead and kernel launches.
* **Memory bandwidth:** eliminating dead nodes and redundant layout operations can reduce unnecessary data movement.
* **Quantization:** lower-precision weights can reduce model size and bandwidth pressure, with accuracy/latency tradeoffs.
* **Backend targeting:** the same graph IR could eventually lower to CPU, GPU, DSP, NPU, or custom accelerator kernels.
* **Correctness:** compiler optimizations must be validated against reference execution before performance claims matter.

## Current limitations

This is a learning project, not a production compiler.

Current limitations:

* No full ONNX importer yet
* No real kernel scheduler
* No hardware-specific backend
* No automatic differentiation
* No advanced graph pattern matching
* Quantization is simulated rather than fully calibrated
* Benchmarks are small and mainly useful for checking the harness

## Future work

* Add a narrow ONNX import path for simple MLPs
* Implement C++/pybind11 kernels for selected operators
* Add operator cost models and pass-ordering experiments
* Add larger benchmark graphs
* Add simple hardware-aware tiling experiments for matrix multiplication
* Compare against ONNX Runtime or PyTorch eager mode on small models
