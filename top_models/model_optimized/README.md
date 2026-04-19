# Optimised Model Engines

TensorRT `.engine`, ONNX `.onnx`, and TorchScript exports of the top-10
checkpoints, optimised for the RTX 4090 (CUDA 13, Compute Capability 8.9).

## Output naming

Every exported file is named `rank{NN}_{name}.{ext}`, sourced from
`top_models/manifest.csv`. This is deterministic and collision-free even
when two upstream checkpoints share a filename (e.g. both YOLO runs
produce `best.pt`).

Examples:

    rank01_yolo26n_320.engine
    rank02_yolo26s_320.onnx
    rank03_mobilenet_selu.onnx
    rank04_mlp_boost.torchscript

## Generating

```bash
cd /home/ubu/Desktop/Assessment

# ONNX only (no TRT required). Uses the legacy tracer (dynamo=False) —
# onnxscript is NOT a dependency.
.venv/bin/python top_models/model_optimized/export_engines.py --format onnx

# TorchScript (traced) — YOLO via ultralytics, MobileNet/MLP via jit.trace.
.venv/bin/python top_models/model_optimized/export_engines.py --format torchscript

# TensorRT engines (FP16 for RTX 4090). Requires `pip install tensorrt`.
.venv/bin/python top_models/model_optimized/export_engines.py --format engine --half

# Everything at once
.venv/bin/python top_models/model_optimized/export_engines.py --format all --half
```

Or via the pipeline:

```bash
.venv/bin/python run_pipeline.py --steps 11 --engine-format all --engine-half
```

## Notes

- YOLO `.engine` files are produced **directly** by Ultralytics. MobileNet and
  MLP `.engine` files must be produced in two steps: export to ONNX first,
  then run `trtexec`. The script prints the exact command.
- Engines are device-specific. An engine built on RTX 4090 will not run on
  RTX 3090 or different CUDA compute capability.
- Engine / ONNX / TorchScript files are large (10 MB – 100 MB+) and are
  excluded from git by `.gitignore`.
- If the YOLO ONNX export fails on a fresh machine, install Ultralytics'
  auto-fetched ONNX deps: `pip install onnx onnxslim`. (MobileNet/MLP ONNX
  uses the legacy tracer and needs only `torch`.)
