# tinyml-compiler-lab

A polished educational project for understanding how tiny neural networks are represented as graphs, optimized by compiler passes, and executed by a runtime.

## What this project demonstrates

- Minimal graph IR (`Tensor`, `Node`, `Graph`)
- Python graph construction path
- Optimization passes:
  - constant folding
  - dead node elimination
  - Linear + ReLU fusion
  - consecutive transpose cleanup
  - int8 weight quantization simulation for linear layers
- NumPy runtime for execution
- Correctness tests for passes + optimized vs unoptimized equivalence
- Benchmarking harness with latency and memory estimates

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
pytest -q
python benchmarks/run_benchmark.py
```

## Compiler pipeline basics

1. **Frontend/build step**: Build a graph in Python (`tinyml_compiler.builder.mlp_graph`).
2. **Midend optimization**: Run passes from `tinyml_compiler.passes.optimize`.
3. **Backend/runtime**: Execute nodes in topological order through NumPy kernels in `tinyml_compiler.runtime`.

## Qualcomm-relevant systems perspective

For edge/SoC deployment thinking:
- **Latency**: Fusion (Linear+ReLU) and dead-node cleanup reduce kernel launches and overhead.
- **Memory bandwidth**: Lowering redundant transpose/layout operations reduces data movement.
- **Quantization**: Simulated int8 weights show model-size and bandwidth benefits and accuracy/latency trade-offs.
- **Accelerator-aware planning**: Separate graph IR + passes from runtime kernels so you can later target CPU, DSP, NPU, or GPU backends with different lowering decisions.

## Optional extension ideas

- Add ONNX import for a narrow supported op subset.
- Implement a pybind11 C++ matmul kernel.
- Add operator cost model + pass ordering experiments.
