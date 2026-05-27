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
- Synthetic benchmark sweep over multiple MLP-style graph sizes

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
python benchmarks/run_sweep.py
````

Expected test result:

```text
5 passed
```

Example single-graph benchmark output:

```text
| Variant   | Latency (ms) | Memory (KB) |
|-----------|-------------:|------------:|
| Baseline  |       0.0650 |       96.62 |
| Optimized |       0.0642 |       96.62 |
```

Example benchmark sweep output:

```text
| Case   | Shape                | Nodes Base -> Opt | Baseline ms | Optimized ms | Speedup | Memory Base KB | Memory Opt KB | Max Abs Error |
|--------|----------------------|------------------:|------------:|-------------:|--------:|---------------:|--------------:|--------------:|
| tiny   | 32x64->128->32       |            4 -> 3 |      0.1150 |       0.1101 |   1.04x |          96.62 |         96.62 |      0.00e+00 |
| small  | 64x256->512->128     |            4 -> 3 |      0.5711 |       0.2946 |   1.94x |        1154.50 |       1154.50 |      0.00e+00 |
| medium | 128x512->2048->512   |            4 -> 3 |      4.4462 |       5.6128 |   0.79x |       11018.00 |      11018.00 |      0.00e+00 |
| large  | 128x1024->4096->1024 |            4 -> 3 |     14.8306 |      13.4840 |   1.10x |       38420.00 |      38420.00 |      0.00e+00 |
```

The benchmark graphs are synthetic MLP-style graphs, not trained models. They are intended to stress the graph representation, optimization pipeline, runtime, and correctness checks.

The latency results should not be interpreted as production performance claims. Since execution is still NumPy-backed, graph-level fusion does not necessarily behave like true low-level kernel fusion. The important result is that the optimizer changes graph structure while preserving numerical outputs.

## Compiler pipeline

1. **Frontend / graph construction**

   Build a neural-network-style graph in Python.

2. **Optimization**

   Run graph passes such as constant folding, dead-node elimination, operator fusion, and transpose cleanup.

3. **Runtime execution**

   Execute the graph using NumPy reference kernels.

4. **Testing and benchmarking**

   Compare optimized and unoptimized outputs for numerical correctness, then measure latency and estimated memory usage.

## Example graph

The benchmark uses synthetic MLP-style graphs such as:

```text
x
↓
Linear
↓
ReLU
↓
Linear
↓
Softmax
↓
prob
```

After optimization, the first `Linear -> ReLU` pair can be fused:

```text
x
↓
FusedLinearReLU
↓
Linear
↓
Softmax
↓
prob
```

This reduces the active graph from 4 nodes to 3 nodes while preserving the final output.

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

## What I learned

One important design issue was separating exact graph optimizations from approximate inference transforms.

Initially, the default optimizer included int8 weight quantization. Correctness tests caught output drift between the baseline graph and optimized graph, especially after softmax. I refactored the optimization API so that default optimization only runs semantics-preserving passes, while quantization is explicitly opt-in.

This distinction matters in real ML systems: not every transformation has the same contract. Some passes should preserve outputs, while others trade numerical precision for memory, latency, or deployment efficiency.

## Current limitations

This is a learning project, not a production compiler.

Current limitations:

* No full ONNX importer yet
* No real kernel scheduler
* No hardware-specific backend
* No automatic differentiation
* No advanced graph pattern matching
* Quantization is simulated rather than fully calibrated
* Benchmarks are synthetic and mainly useful for checking the compiler/runtime pipeline
* Fusion is graph-level only; it does not yet lower to custom fused C++/hardware kernels

## Future work

* Add a narrow ONNX import path for simple MLPs
* Implement C++/pybind11 kernels for selected operators
* Add real low-level fused kernels for `Linear + ReLU`
* Improve memory estimation to count only active tensors during optimized execution
* Add operator cost models and pass-ordering experiments
* Add larger benchmark graphs
* Add simple hardware-aware tiling experiments for matrix multiplication
* Compare against ONNX Runtime or PyTorch eager mode on small models
