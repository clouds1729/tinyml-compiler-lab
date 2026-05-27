# Compiler Passes

## 1) Constant folding
If a node's inputs are all compile-time constants, execute it at compile time and replace output tensors with constants.

## 2) Dead node elimination
Walk backward from graph outputs and keep only nodes that contribute to requested outputs.

## 3) Linear + ReLU fusion
Replace `Linear -> ReLU` with `FusedLinearReLU` to reduce intermediate writes and runtime overhead.

## 4) Consecutive transpose cleanup
Collapse transpose chains when composition resolves to identity.

## 5) Simulated int8 quantization (weights)
For linear weights, compute scale using max-abs calibration and store int8 tensors + scale metadata.
Runtime dequantizes to float for reference execution.
