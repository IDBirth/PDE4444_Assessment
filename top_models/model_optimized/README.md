# Optimised Model Engines

This folder holds TensorRT `.engine` and ONNX `.onnx` exports of the top-10 checkpoints,
optimised for the RTX 4090 (CUDA 13, Compute Capability 8.9).

## Current status

| File | Status |
|---|---|
| `*.engine` | Not yet generated — TensorRT Python bindings not in venv |
| `*.onnx`   | Generate with `export_engines.py --format onnx` |

## Generating

```bash
cd /home/ubu/Desktop/Assessment

# ONNX only (no TRT required)
.venv/bin/python top_models/model_optimized/export_engines.py --format onnx

# TensorRT engines (FP16 for RTX 4090) — requires TRT installed
pip install tensorrt
.venv/bin/python top_models/model_optimized/export_engines.py --format engine --half

# Both
.venv/bin/python top_models/model_optimized/export_engines.py --format all --half
```

## Using .engine in the demo

Pass the `.engine` path to the demo script as `--models` — the YOLO engine is loaded by
Ultralytics automatically and runs 2-4× faster than the .pt checkpoint at FP16.

## Notes

- Engines are device-specific. An engine built on RTX 4090 will not run on RTX 3090 or different CUDA compute capability.
- MobileNetV2 and MLP engines are ONNX → TRT via `trtexec` (see export script).
- Engine files are large (100 MB+) and should not be committed to git. Add to `.gitignore` if needed.
