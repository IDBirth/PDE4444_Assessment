#!/usr/bin/env python3
"""
Export top-10 model checkpoints to optimised inference formats.

Output naming is deterministic and collision-free:
    rank{NN}_{name}.{ext}

    e.g. rank01_yolo26n_320.engine
         rank02_yolo26s_320.onnx
         rank03_mobilenet_selu.onnx
         rank04_mlp_boost.torchscript

Formats:
    engine       TensorRT .engine (YOLO direct; MobileNet/MLP via trtexec hint)
    onnx         ONNX .onnx
    torchscript  TorchScript (YOLO: via ultralytics; PyTorch: via jit.trace)
    all          all of the above

Usage:
    cd /home/ubu/Desktop/Assessment
    .venv/bin/python top_models/model_optimized/export_engines.py \
        [--format engine|onnx|torchscript|all] \
        [--device 0]          # GPU index for TRT/ONNX export
        [--half]              # FP16 (recommended for RTX 4090)

Requirements:
    pip install tensorrt     # for .engine (YOLO direct)
    # ONNX export uses the legacy tracer (dynamo=False) so onnxscript
    # is NOT required.
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT   = Path(__file__).resolve().parent.parent.parent
OUT_DIR     = Path(__file__).resolve().parent
MANIFEST    = REPO_ROOT / "top_models" / "manifest.csv"

YOLO_IMGSZ_DEFAULT = 320


# ── helpers ──────────────────────────────────────────────────────────────────

def _out_stem(entry: dict) -> str:
    """rank{NN}_{name} — zero-padded so filenames sort naturally."""
    rank = str(entry["rank"]).zfill(2)
    name = entry["name"].strip().replace(" ", "_")
    return f"rank{rank}_{name}"


def _move_to_out(src: Path, dst_stem: str, fmt: str) -> Path:
    ext_map = {"engine": ".engine", "onnx": ".onnx", "torchscript": ".torchscript"}
    ext = ext_map.get(fmt, src.suffix)
    dst = OUT_DIR / f"{dst_stem}{ext}"
    if dst.exists():
        dst.unlink()
    shutil.move(str(src), str(dst))
    return dst


# ── YOLO ─────────────────────────────────────────────────────────────────────

def export_yolo(ckpt: Path, fmt: str, device: str, half: bool,
                imgsz: int, out_stem: str) -> None:
    from ultralytics import YOLO
    model = YOLO(str(ckpt))
    kwargs: dict = {"format": fmt, "imgsz": imgsz, "device": device}
    # half only applies to engine/onnx for YOLO
    if half and fmt in {"engine", "onnx"}:
        kwargs["half"] = True
    print(f"  Exporting {ckpt.parent.name}/{ckpt.name} -> {fmt} "
          f"(device={device}, half={half}) ...")
    exported = model.export(**kwargs)
    if not exported:
        raise RuntimeError(f"YOLO export returned empty path for {ckpt}")
    dst = _move_to_out(Path(exported), out_stem, fmt)
    print(f"  -> {dst}")


# ── PyTorch model builders (same shapes as training) ────────────────────────

def _build_mobilenet(ckpt: Path):
    import torch
    import torch.nn as nn
    from torchvision import models

    act_map = {"relu": nn.ReLU, "elu": nn.ELU, "gelu": nn.GELU,
               "selu": nn.SELU, "leaky_relu": nn.LeakyReLU}
    act_name = "relu"
    for p in ckpt.parts:
        for k in act_map:
            if k in p.lower():
                act_name = k
    backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    backbone.classifier = nn.Sequential(
        nn.Linear(backbone.last_channel, 256),
        nn.BatchNorm1d(256),
        act_map[act_name](),
        nn.Dropout(0.3),
        nn.Linear(256, 1),
    )
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    backbone.load_state_dict(state, strict=True)
    backbone.eval()
    # Match training: fine-tuned runs used 256, frozen runs used 224.
    full_path_lower = "/".join(p.lower() for p in ckpt.parts)
    img_size = 256 if "finetune" in full_path_lower else 224
    return backbone, img_size


def _build_mlp(ckpt: Path):
    import torch
    import torch.nn as nn

    _ACT = {"relu": nn.ReLU, "elu": nn.ELU, "gelu": nn.GELU,
            "selu": nn.SELU, "leaky_relu": nn.LeakyReLU}

    state     = torch.load(ckpt, map_location="cpu", weights_only=True)
    input_dim = int(state["net.1.weight"].shape[0])
    img_size  = round((input_dim / 3) ** 0.5)

    hp_path = ckpt.parent / "best_hyperparameters.json"
    if hp_path.exists():
        hp     = json.loads(hp_path.read_text())
        hidden = ast.literal_eval(hp["hidden_sizes"])
        act    = hp["activation"]
        drop   = float(hp["dropout"])
    else:
        hidden, act, drop = (256,), "elu", 0.4

    layers = [nn.Flatten(), nn.BatchNorm1d(input_dim)]
    in_d = input_dim
    for h in hidden:
        layers += [nn.Linear(in_d, h), _ACT[act]()]
        if drop > 0:
            layers.append(nn.Dropout(drop))
        in_d = h
    layers.append(nn.Linear(in_d, 1))

    class _MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(*layers)
        def forward(self, x):
            return self.net(x).squeeze(1)

    model = _MLP()
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, img_size


# ── PyTorch export paths ─────────────────────────────────────────────────────

def export_torch_onnx(ckpt: Path, model_type: str, out_stem: str) -> Path:
    import torch

    if model_type == "mobilenet":
        model, img_size = _build_mobilenet(ckpt)
    else:
        model, img_size = _build_mlp(ckpt)

    dst   = OUT_DIR / f"{out_stem}.onnx"
    dummy = torch.randn(1, 3, img_size, img_size)

    # dynamo=False forces the legacy tracer, which does NOT depend on onnxscript.
    # Fall back silently on older torch versions that don't expose the kwarg.
    try:
        torch.onnx.export(
            model, dummy, str(dst),
            input_names=["image"], output_names=["logit"],
            opset_version=17,
            dynamic_axes={"image": {0: "batch"}},
            dynamo=False,
        )
    except TypeError:
        torch.onnx.export(
            model, dummy, str(dst),
            input_names=["image"], output_names=["logit"],
            opset_version=17,
            dynamic_axes={"image": {0: "batch"}},
        )
    print(f"  -> {dst}")
    return dst


def export_torch_torchscript(ckpt: Path, model_type: str, out_stem: str) -> Path:
    import torch

    if model_type == "mobilenet":
        model, img_size = _build_mobilenet(ckpt)
    else:
        model, img_size = _build_mlp(ckpt)

    dst   = OUT_DIR / f"{out_stem}.torchscript"
    dummy = torch.randn(1, 3, img_size, img_size)
    traced = torch.jit.trace(model, dummy)
    traced.save(str(dst))
    print(f"  -> {dst}")
    return dst


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--format",  default="all",
                        choices=["engine", "onnx", "torchscript", "all"])
    parser.add_argument("--device",  default="0", help="GPU index (default 0)")
    parser.add_argument("--half",    action="store_true", help="FP16 for TRT/ONNX")
    args = parser.parse_args()

    do_engine      = args.format in {"engine", "all"}
    do_onnx        = args.format in {"onnx", "all"}
    do_torchscript = args.format in {"torchscript", "all"}

    has_trt = False
    if do_engine:
        try:
            import tensorrt
            has_trt = True
            print(f"[INFO] TensorRT {tensorrt.__version__} found.")
        except ImportError:
            print("[WARN] TensorRT not installed — .engine skipped. "
                  "Install with: pip install tensorrt")

    with MANIFEST.open(newline="") as fh:
        rows = list(csv.DictReader(fh))

    produced: list[Path] = []
    failures: list[str]  = []

    for entry in rows:
        ckpt = REPO_ROOT / entry["checkpoint_path"]
        if not ckpt.exists():
            print(f"[SKIP] {entry['name']} — checkpoint not found: {ckpt}")
            continue

        model_type = entry["model_type"].replace("_finetuned", "")
        out_stem   = _out_stem(entry)
        print(f"\n[rank {entry['rank']}] {entry['name']}  ({model_type})  "
              f"out_stem={out_stem}")

        if model_type == "yolo":
            try:
                imgsz = int(entry.get("img_size") or YOLO_IMGSZ_DEFAULT)
            except ValueError:
                imgsz = YOLO_IMGSZ_DEFAULT

            if do_engine and has_trt:
                try:
                    export_yolo(ckpt, "engine", args.device, args.half, imgsz, out_stem)
                    produced.append(OUT_DIR / f"{out_stem}.engine")
                except Exception as exc:
                    failures.append(f"{out_stem} engine: {exc}")
                    print(f"  [FAIL] engine: {exc}")
            if do_onnx:
                try:
                    export_yolo(ckpt, "onnx", args.device, args.half, imgsz, out_stem)
                    produced.append(OUT_DIR / f"{out_stem}.onnx")
                except Exception as exc:
                    failures.append(f"{out_stem} onnx: {exc}")
                    print(f"  [FAIL] onnx: {exc}")
            if do_torchscript:
                try:
                    export_yolo(ckpt, "torchscript", args.device, False,
                                imgsz, out_stem)
                    produced.append(OUT_DIR / f"{out_stem}.torchscript")
                except Exception as exc:
                    failures.append(f"{out_stem} torchscript: {exc}")
                    print(f"  [FAIL] torchscript: {exc}")

        else:  # mobilenet / mlp
            if do_onnx:
                try:
                    dst = export_torch_onnx(ckpt, model_type, out_stem)
                    produced.append(dst)
                except Exception as exc:
                    failures.append(f"{out_stem} onnx: {exc}")
                    print(f"  [FAIL] onnx: {exc}")
            if do_torchscript:
                try:
                    dst = export_torch_torchscript(ckpt, model_type, out_stem)
                    produced.append(dst)
                except Exception as exc:
                    failures.append(f"{out_stem} torchscript: {exc}")
                    print(f"  [FAIL] torchscript: {exc}")
            if do_engine and has_trt:
                onnx_file   = OUT_DIR / f"{out_stem}.onnx"
                engine_file = OUT_DIR / f"{out_stem}.engine"
                if onnx_file.exists():
                    cmd = (f"trtexec --onnx={onnx_file} "
                           f"--saveEngine={engine_file}")
                    if args.half:
                        cmd += " --fp16"
                    print(f"  [INFO] build TRT engine with:\n    {cmd}")
                else:
                    print(f"  [INFO] build TRT engine: run --format onnx first, "
                          f"then trtexec on {out_stem}.onnx")

    print("\n" + "="*70)
    print(f"Done. Produced {len(produced)} file(s) in: {OUT_DIR}")
    for p in produced:
        print(f"  {p.name}")
    if failures:
        print(f"\n{len(failures)} failure(s):")
        for f in failures:
            print(f"  x {f}")


if __name__ == "__main__":
    main()
