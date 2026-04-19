#!/usr/bin/env python3
# Runs top-5 model checkpoints simultaneously and shows 6 panels (5 models + 1 ensemble).
# No auto-capture.  Asymmetric thresholds: PASS ≤ 0.40 defect-prob, FAIL ≥ 0.70, else UNCERTAIN.
#
# Example usage (stream):
#   python defect_classification_stack/demo_live_test_with_video_top6.py \
#       --stream-url "https://api.baslarlabs.ae/video_feed" \
#       --auth-user zeroq --auth-password '#@45459xQw'
#
# Offline:
#   python defect_classification_stack/demo_live_test_with_video_top6.py \
#       --input path/to/images_or_video
#
# Override default top-5 paths:
#   python defect_classification_stack/demo_live_test_with_video_top6.py \
#       --models model1.pt model2.pt model3.pt model4.pt model5.pt \
#       --input path/to/test
from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import re
import statistics
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Iterator, Sequence
from urllib.parse import unquote, urljoin, urlsplit, urlunsplit

import cv2
import numpy as np
import requests
import torch
import torch.nn as nn
import urllib3
from torchvision import models, transforms
from ultralytics import YOLO

REPO_ROOT   = Path(__file__).resolve().parent.parent
STACK_ROOT  = Path(__file__).resolve().parent
DEFAULT_DEMO_DIR = STACK_ROOT / "demo_sessions"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ── Asymmetric thresholds (user-tunable via CLI) ────────────────────────────
DEFECT_THRESHOLD_PASS = 0.40   # defect_prob ≤ this  →  PASS
DEFECT_THRESHOLD_FAIL = 0.70   # defect_prob ≥ this  →  FAIL
# gap (0.40, 0.70) → UNCERTAIN

GT_HOTKEY_MAP = {ord("1"): "PASS", ord("2"): "FAIL"}

DEVICE = torch.device(
    "mps"  if torch.backends.mps.is_available()  else
    "cuda" if torch.cuda.is_available()           else
    "cpu"
)

# ── Default top-5 checkpoints (from top10_models.md ranks 1-5) ──────────────
DEFAULT_MODEL_PATHS: list[Path] = [
    REPO_ROOT / "runs3/iter1_yolo_320/train/weights/best.pt",        # rank 1 YOLO26n-320
    REPO_ROOT / "runs3/iter2_yolo_s/train/weights/best.pt",          # rank 2 YOLO26s-320
    REPO_ROOT / "runs3/iter3_mobilenet_finetune/selu/model.pt",       # rank 3 MNet-SELU r3
    REPO_ROOT / "runs3/iter1_mlp_boost/best_model.pt",               # rank 4 MLP-boost r3
    STACK_ROOT / "runs/iter3_mobilenet_finetune/elu/model.pt",        # rank 5 MNet-ELU r1
]
DEFAULT_DISPLAY_NAMES: list[str] = [
    "YOLO26n-320",
    "YOLO26s-320",
    "MNet-SELU(r3)",
    "MLP-boost(r3)",
    "MNet-ELU(r1)",
]

_ACTIVATION_MAP: dict[str, type[nn.Module]] = {
    "relu":       nn.ReLU,
    "elu":        nn.ELU,
    "gelu":       nn.GELU,
    "selu":       nn.SELU,
    "leaky_relu": nn.LeakyReLU,
}


# ── Data structures ──────────────────────────────────────────────────────────

def env_or_none(name: str) -> str | None:
    value = os.getenv(name)
    return value if value else None


@dataclass
class FramePacket:
    frame_index: int
    frame: np.ndarray
    source_label: str


@dataclass
class ModelPrediction:
    model_name:   str
    model_type:   str
    class_name:   str
    decision:     str       # PASS | FAIL | UNCERTAIN
    confidence:   float     # max(probabilities)
    latency_ms:   float
    probabilities: dict[str, float]
    defect_prob:  float     # P(defect) — used for ensemble mean


# ── Defect probability extraction ────────────────────────────────────────────

def get_defect_prob(probabilities: dict[str, float]) -> float:
    """Return P(defect) regardless of class naming convention."""
    for label, prob in probabilities.items():
        norm = label.strip().lower()
        if "def" in norm and "non" not in norm:
            return prob
    return max(probabilities.values())


def threshold_decision(defect_prob: float, pass_thr: float, fail_thr: float) -> str:
    if defect_prob <= pass_thr:
        return "PASS"
    if defect_prob >= fail_thr:
        return "FAIL"
    return "UNCERTAIN"


# ── MLP model (mirrors train_keras_mlp_random_search.MLP) ───────────────────

class MLPModel(nn.Module):
    def __init__(self, input_dim: int, hidden_sizes: tuple[int, ...], activation: str, dropout: float) -> None:
        super().__init__()
        act_cls = _ACTIVATION_MAP[activation]
        layers: list[nn.Module] = [nn.Flatten(), nn.BatchNorm1d(input_dim)]
        in_dim = input_dim
        for h in hidden_sizes:
            layers += [nn.Linear(in_dim, h), act_cls()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


def build_mlp_model_from_checkpoint(checkpoint_path: Path) -> tuple[MLPModel, int]:
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    # Recover input_dim from the BatchNorm1d weight (net.1)
    input_dim = int(state["net.1.weight"].shape[0])
    img_size  = round((input_dim / 3) ** 0.5)  # e.g. 27648/3=9216 → √9216=96

    hp_path = checkpoint_path.parent / "best_hyperparameters.json"
    if hp_path.exists():
        hp = json.loads(hp_path.read_text())
        hidden_sizes: tuple[int, ...] = ast.literal_eval(hp["hidden_sizes"])
        activation = hp["activation"]
        dropout    = float(hp["dropout"])
    else:
        # runs1 MLP fallback
        hidden_sizes = (256,)
        activation   = "elu"
        dropout      = 0.4

    model = MLPModel(input_dim, hidden_sizes, activation, dropout)
    model.load_state_dict(state, strict=True)
    model.to(DEVICE)
    model.eval()
    return model, img_size


def build_mlp_transform(img_size: int) -> transforms.Compose:
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


# ── MobileNet helpers ────────────────────────────────────────────────────────

def build_mobilenet_transform() -> transforms.Compose:
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def build_mobilenet_model(activation_name: str, checkpoint_path: Path) -> nn.Module:
    backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    act_cls  = _ACTIVATION_MAP[activation_name]
    backbone.classifier = nn.Sequential(
        nn.Linear(backbone.last_channel, 256),
        nn.BatchNorm1d(256),
        act_cls(),
        nn.Dropout(0.3),
        nn.Linear(256, 1),
    )
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    backbone.load_state_dict(state, strict=True)
    backbone.to(DEVICE)
    backbone.eval()
    return backbone


def detect_activation_from_path(path: Path) -> str:
    lower_parts = [part.lower() for part in path.parts]
    for activation in ["leaky_relu", "relu", "elu", "gelu", "selu"]:
        if any(activation in part for part in lower_parts):
            return activation
    raise ValueError(
        f"Could not infer MobileNet activation from path: {path}. "
        "Expected one of relu/elu/gelu/selu/leaky_relu in the path."
    )


def infer_class_names_from_report(path: Path) -> list[str]:
    report_path = path.parent / "classification_report.csv"
    if not report_path.exists():
        return ["non_defect", "defect"]
    rows: list[str] = []
    with report_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        next(reader, None)
        for row in reader:
            if not row:
                continue
            label = row[0].strip()
            if label in {"accuracy", "macro avg", "weighted avg"}:
                continue
            if label:
                rows.append(label)
            if len(rows) == 2:
                break
    return rows if len(rows) == 2 else ["non_defect", "defect"]


def infer_class_names_for_model(path: Path, model_type: str) -> list[str]:
    if model_type in {"mobilenet", "mlp"}:
        return infer_class_names_from_report(path)
    return ["defective", "non_defective"]  # YOLO default


# ── Model type detection ─────────────────────────────────────────────────────

def detect_model_type(path: Path) -> str:
    try:
        m = YOLO(str(path))
        if getattr(m, "task", None) == "classify":
            return "yolo"
    except Exception:
        pass
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError(f"Could not load checkpoint {path}: {exc}") from exc
    if isinstance(state, dict):
        keys = [str(k) for k in state]
        if any(k.startswith("net.") for k in keys):
            return "mlp"
        if any(k.startswith("features.") for k in keys):
            return "mobilenet"
    raise ValueError(f"Unsupported checkpoint format: {path}")


# ── DemoModel ────────────────────────────────────────────────────────────────

class DemoModel:
    def __init__(
        self,
        model_path: Path,
        display_name: str | None = None,
        pass_threshold: float = DEFECT_THRESHOLD_PASS,
        fail_threshold: float = DEFECT_THRESHOLD_FAIL,
    ) -> None:
        self.path           = model_path
        self.display_name   = display_name or model_path.stem
        self.pass_threshold = pass_threshold
        self.fail_threshold = fail_threshold
        self.model_type     = detect_model_type(model_path)
        self.class_names    = infer_class_names_for_model(model_path, self.model_type)

        self._yolo:      YOLO | None                = None
        self._mobilenet: nn.Module | None           = None
        self._mlp:       MLPModel | None            = None
        self._transform: transforms.Compose | None  = None
        self._mlp_img_size: int                     = 96

        if self.model_type == "yolo":
            self._yolo = YOLO(str(model_path))
        elif self.model_type == "mobilenet":
            act = detect_activation_from_path(model_path)
            self.class_names  = infer_class_names_from_report(model_path)
            self._mobilenet   = build_mobilenet_model(act, model_path)
            self._transform   = build_mobilenet_transform()
        elif self.model_type == "mlp":
            self.class_names      = infer_class_names_from_report(model_path)
            self._mlp, self._mlp_img_size = build_mlp_model_from_checkpoint(model_path)
            self._transform = build_mlp_transform(self._mlp_img_size)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def predict(self, frame: np.ndarray) -> ModelPrediction:
        start = perf_counter()

        if self.model_type == "yolo":
            assert self._yolo is not None
            result      = self._yolo.predict(source=frame, verbose=False)[0]
            probs_tensor = result.probs.data.detach().cpu().float()
            top_index   = int(result.probs.top1)
            class_name  = str(result.names[top_index])
            probabilities = {
                str(result.names[i]): float(p.item())
                for i, p in enumerate(probs_tensor)
            }

        else:
            # MobileNet or MLP — both output a single logit
            assert self._transform is not None
            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tensor = self._transform(rgb).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                if self.model_type == "mobilenet":
                    assert self._mobilenet is not None
                    logit = self._mobilenet(tensor).squeeze(0)
                else:
                    assert self._mlp is not None
                    logit = self._mlp(tensor).squeeze(0)

            positive_prob = float(torch.sigmoid(logit).item())
            negative_label, positive_label = self.class_names
            probabilities = {
                negative_label: 1.0 - positive_prob,
                positive_label: positive_prob,
            }
            class_name = positive_label if positive_prob >= 0.5 else negative_label

        latency_ms  = (perf_counter() - start) * 1000.0
        defect_prob = get_defect_prob(probabilities)
        decision    = threshold_decision(defect_prob, self.pass_threshold, self.fail_threshold)

        return ModelPrediction(
            model_name=self.display_name,
            model_type=self.model_type,
            class_name=class_name,
            decision=decision,
            confidence=max(probabilities.values()),
            latency_ms=latency_ms,
            probabilities=probabilities,
            defect_prob=defect_prob,
        )


# ── Ensemble ─────────────────────────────────────────────────────────────────

def compute_ensemble(
    predictions: list[ModelPrediction],
    pass_threshold: float,
    fail_threshold: float,
) -> ModelPrediction:
    mean_defect = sum(p.defect_prob for p in predictions) / len(predictions)
    decision    = threshold_decision(mean_defect, pass_threshold, fail_threshold)
    class_name  = "defect" if mean_defect >= 0.5 else "non_defect"
    return ModelPrediction(
        model_name=f"ENSEMBLE ({len(predictions)} models)",
        model_type="ensemble",
        class_name=class_name,
        decision=decision,
        confidence=mean_defect if class_name == "defect" else 1.0 - mean_defect,
        latency_ms=0.0,
        probabilities={"non_defect": 1.0 - mean_defect, "defect": mean_defect},
        defect_prob=mean_defect,
    )


# ── Stream / input iterators (unchanged from original) ──────────────────────

def sanitize_stream_url(url: str) -> str:
    parts = urlsplit(url)
    host  = parts.hostname or ""
    if parts.port:
        host = f"{host}:{parts.port}"
    return urlunsplit((parts.scheme, host, parts.path, parts.query, parts.fragment))


def resolve_basic_auth(
    stream_url: str,
    auth_mode: str,
    auth_user: str | None,
    auth_password: str | None,
) -> tuple[str, tuple[str, str] | None]:
    parts         = urlsplit(stream_url)
    sanitized_url = sanitize_stream_url(stream_url)
    if auth_mode == "none":
        return sanitized_url, None
    user     = auth_user
    password = auth_password
    if auth_mode in {"auto", "basic", "form"}:
        if user is None and parts.username is not None:
            user = unquote(parts.username)
        if password is None and parts.password is not None:
            password = unquote(parts.password)
    if auth_mode in {"basic", "form"}:
        if user is None or password is None:
            raise ValueError("Auth mode requires --auth-user / --auth-password or credentials in the URL.")
    elif auth_mode == "auto" and (user is None or password is None):
        return sanitized_url, None
    return sanitized_url, (user or "", password or "")


def is_login_page(content_type: str, text: str) -> bool:
    if "html" not in content_type.lower():
        return False
    lower = text.lower()
    return "login" in lower and "username" in lower and "password" in lower and "<form" in lower


def extract_login_form_details(html: str, base_url: str) -> tuple[str, dict[str, str]]:
    action_match = re.search(r"<form[^>]*action=[\"']([^\"']+)[\"']", html, flags=re.IGNORECASE)
    action_url   = urljoin(base_url, action_match.group(1)) if action_match else base_url
    payload: dict[str, str] = {}
    hidden_pat = re.compile(
        r"<input[^>]*type=[\"']hidden[\"'][^>]*name=[\"']([^\"']+)[\"'][^>]*value=[\"']([^\"']*)[\"']",
        flags=re.IGNORECASE,
    )
    for m in hidden_pat.finditer(html):
        payload[m.group(1)] = m.group(2)
    return action_url, payload


def login_via_form(
    session: requests.Session,
    login_page_response: requests.Response,
    auth: tuple[str, str],
    login_url_override: str | None,
    verify_tls: bool,
    timeout: float,
) -> None:
    login_url, payload = extract_login_form_details(login_page_response.text, login_page_response.url)
    if login_url_override:
        login_url = login_url_override
    payload["username"] = auth[0]
    payload["password"] = auth[1]
    print(f"[INFO] HTML login form detected, posting to: {login_url}")
    resp = session.post(
        login_url, data=payload, timeout=timeout, verify=verify_tls,
        headers={"Referer": login_page_response.url}, allow_redirects=True,
    )
    resp.raise_for_status()
    print(f"[INFO] login POST HTTP {resp.status_code}, final URL={resp.url}")


def open_stream_response(
    session: requests.Session,
    stream_url: str,
    auth_mode: str,
    auth: tuple[str, str] | None,
    login_url: str | None,
    verify_tls: bool,
    timeout: float,
) -> requests.Response:
    request_auth = auth if auth_mode in {"auto", "basic"} else None
    response = session.get(stream_url, stream=True, auth=request_auth, timeout=(timeout, None), verify=verify_tls)
    response.raise_for_status()
    content_type = response.headers.get("content-type", "<missing>")
    if auth_mode in {"auto", "form"} and auth is not None and is_login_page(content_type, response.text):
        login_via_form(
            session=session, login_page_response=response, auth=auth,
            login_url_override=login_url, verify_tls=verify_tls, timeout=timeout,
        )
        response.close()
        response = session.get(stream_url, stream=True, timeout=(timeout, None), verify=verify_tls)
        response.raise_for_status()
    return response


def iter_mjpeg_frames(
    stream_url: str,
    auth_mode: str,
    auth: tuple[str, str] | None,
    login_url: str | None,
    verify_tls: bool,
    timeout: float,
    debug_bytes: int,
) -> Iterator[FramePacket]:
    with requests.Session() as session:
        with open_stream_response(
            session=session, stream_url=stream_url, auth_mode=auth_mode,
            auth=auth, login_url=login_url, verify_tls=verify_tls, timeout=timeout,
        ) as response:
            response.raise_for_status()
            content_type = response.headers.get("content-type", "<missing>")
            print(f"[INFO] stream HTTP {response.status_code}, content-type={content_type}")
            buffer    = bytearray()
            saw_frame = False
            frame_index = 0
            for chunk in response.iter_content(chunk_size=8192):
                if not chunk:
                    continue
                buffer.extend(chunk)
                while True:
                    start = buffer.find(b"\xff\xd8")
                    end   = buffer.find(b"\xff\xd9", start + 2 if start != -1 else 0)
                    if start == -1 or end == -1:
                        if start > 0:
                            del buffer[:start]
                        break
                    jpeg_bytes = bytes(buffer[start : end + 2])
                    del buffer[: end + 2]
                    frame = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if frame is not None:
                        saw_frame = True
                        frame_index += 1
                        yield FramePacket(frame_index=frame_index, frame=frame, source_label="stream")
            if not saw_frame:
                preview = bytes(buffer[:debug_bytes]).decode("utf-8", errors="replace")
                raise RuntimeError(
                    f"No JPEG frames decoded. content-type={content_type!r}. Preview: {preview!r}"
                )


def iter_input_frames(path: Path) -> Iterator[FramePacket]:
    if path.is_dir():
        image_paths = sorted([p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS])
        if not image_paths:
            raise FileNotFoundError(f"No images found in: {path}")
        for idx, ip in enumerate(image_paths, start=1):
            frame = cv2.imread(str(ip))
            if frame is not None:
                yield FramePacket(frame_index=idx, frame=frame, source_label=ip.name)
        return
    if path.suffix.lower() in IMAGE_EXTENSIONS:
        frame = cv2.imread(str(path))
        if frame is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        yield FramePacket(frame_index=1, frame=frame, source_label=path.name)
        return
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            idx += 1
            yield FramePacket(frame_index=idx, frame=frame, source_label=path.name)
    finally:
        cap.release()


# ── Display helpers ──────────────────────────────────────────────────────────

def fit_for_display(frame: np.ndarray, max_width: int, max_height: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if w <= max_width and h <= max_height:
        return frame
    scale    = min(max_width / w, max_height / h)
    new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
    return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)


def prepare_display_window(window_name: str, width: int, height: int) -> None:
    flags = cv2.WINDOW_NORMAL
    if hasattr(cv2, "WINDOW_KEEPRATIO"):
        flags |= cv2.WINDOW_KEEPRATIO
    cv2.namedWindow(window_name, flags)
    cv2.resizeWindow(window_name, width, height)


def draw_probability_bar(
    frame: np.ndarray, x: int, y: int, width: int, height: int,
    fraction: float, fill_color: tuple[int, int, int],
) -> None:
    cv2.rectangle(frame, (x, y), (x + width, y + height), (80, 80, 80), 1)
    cv2.rectangle(
        frame, (x + 1, y + 1),
        (x + max(1, int((width - 2) * max(0.0, min(1.0, fraction)))), y + height - 1),
        fill_color, -1,
    )


def _decision_color(decision: str) -> tuple[int, int, int]:
    if decision == "FAIL":
        return (0, 0, 255)
    if decision == "PASS":
        return (0, 200, 0)
    return (0, 165, 255)  # UNCERTAIN = orange


def draw_overlay(
    frame: np.ndarray,
    model_predictions: Sequence[ModelPrediction],
    ensemble: ModelPrediction | None,
    frame_index: int,
    fps: float,
    paused: bool,
    source_label: str,
    frame_size_text: str,
    startup_lines: Sequence[str],
    pending_ground_truth: bool,
    last_ground_truth: str | None,
    ensemble_counts: dict[str, int],
    pass_threshold: float,
    fail_threshold: float,
) -> np.ndarray:
    annotated = frame.copy()
    h, w = annotated.shape[:2]

    # ── Layout constants ──────────────────────────────────────────────────
    COLS      = 3
    ROWS      = 2
    gap       = 6
    header_h  = 28
    panel_w   = (w - (COLS + 1) * gap) // COLS
    panel_h   = min(max(85, h // 5), 135)
    title_h   = max(22, panel_h // 5)
    grid_h    = ROWS * panel_h + (ROWS + 1) * gap
    grid_y0   = h - grid_h - gap

    all_panels: list[ModelPrediction] = list(model_predictions)
    if ensemble is not None:
        all_panels.append(ensemble)

    # ── Header strip ──────────────────────────────────────────────────────
    hdr_bg = annotated.copy()
    cv2.rectangle(hdr_bg, (0, 0), (w, header_h), (10, 10, 10), -1)
    cv2.addWeighted(hdr_bg, 0.82, annotated, 0.18, 0, annotated)

    status  = "PAUSED" if paused else "LIVE"
    gt_text = "awaiting 1/2" if pending_ground_truth else (last_ground_truth or "-")
    hdr = (
        f"TOP-6  frame={frame_index}  fps={fps:.1f}  {status}  "
        f"ensemble P/F/U={ensemble_counts['PASS']}/{ensemble_counts['FAIL']}/{ensemble_counts['UNCERTAIN']}  "
        f"thr PASS\u2264{pass_threshold:.2f} FAIL\u2265{fail_threshold:.2f}  "
        f"gt={gt_text}  {frame_size_text}    "
        "SPACE=cap  p=pause  q=quit"
    )
    cv2.putText(annotated, hdr, (8, header_h - 7),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (220, 220, 220), 1, cv2.LINE_AA)

    # ── Grid background ───────────────────────────────────────────────────
    grid_bg = annotated.copy()
    cv2.rectangle(grid_bg, (0, grid_y0 - gap), (w, h), (12, 12, 12), -1)
    cv2.addWeighted(grid_bg, 0.80, annotated, 0.20, 0, annotated)

    # ── Individual panels (3 × 2 grid) ───────────────────────────────────
    for i, pred in enumerate(all_panels):
        row = i // COLS
        col = i % COLS
        x0 = gap + col * (panel_w + gap)
        y0 = grid_y0 + gap + row * (panel_h + gap)
        x1 = x0 + panel_w
        y1 = y0 + panel_h

        color        = _decision_color(pred.decision)
        is_ensemble  = pred.model_type == "ensemble"
        border_thick = 3 if is_ensemble else 2

        # Title bar background
        cv2.rectangle(annotated, (x0, y0), (x1, y0 + title_h), (38, 38, 38), -1)

        # Model name in title bar
        name_label = f"\u25b6 {pred.model_name}" if is_ensemble else pred.model_name
        cv2.putText(
            annotated, name_label,
            (x0 + 5, y0 + title_h - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48 if is_ensemble else 0.42,
            (255, 255, 0) if is_ensemble else (240, 240, 240),
            1, cv2.LINE_AA,
        )

        # Colored perimeter border drawn over title bar so it frames everything
        cv2.rectangle(annotated, (x0, y0), (x1, y1), color, border_thick)
        if is_ensemble:
            # Extra inner rect to make ensemble border more prominent
            cv2.rectangle(annotated, (x0 + 3, y0 + 3), (x1 - 3, y1 - 3), color, 1)

        # Content starts below title bar
        cy = y0 + title_h + 14

        # Decision text
        if is_ensemble:
            cv2.putText(annotated, f"VERDICT: {pred.decision}",
                        (x0 + 5, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.60, color, 2, cv2.LINE_AA)
        else:
            cv2.putText(annotated, pred.decision,
                        (x0 + 5, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 1, cv2.LINE_AA)
        cy += 17

        # defect_prob value
        prob_label = f"mean_defect={pred.defect_prob:.3f}" if is_ensemble else f"defect={pred.defect_prob:.3f}"
        cv2.putText(annotated, prob_label,
                    (x0 + 5, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (210, 210, 210), 1, cv2.LINE_AA)
        cy += 13

        # Probability bar filling most of panel width
        bar_w = panel_w - 12
        draw_probability_bar(annotated, x0 + 5, cy - 7, bar_w, 9, pred.defect_prob, color)
        cy += 11

        # Model type + latency (skip for ensemble)
        if not is_ensemble:
            cv2.putText(
                annotated,
                f"[{pred.model_type}]  {pred.latency_ms:.0f}ms",
                (x0 + 5, cy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (140, 140, 140), 1, cv2.LINE_AA,
            )

    return annotated


# ── Capture / session helpers (adapted) ──────────────────────────────────────

def save_capture(
    capture_dir: Path, annotated_frame: np.ndarray, frame_index: int, reason: str,
) -> Path:
    capture_dir.mkdir(parents=True, exist_ok=True)
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{reason}_{ts}_f{frame_index:06d}.jpg"
    out      = capture_dir / filename
    cv2.imwrite(str(out), annotated_frame)
    return out


def session_summary(events: Sequence[dict]) -> dict:
    if not events:
        return {"total_events": 0, "per_model": {}}
    per_model: dict[str, dict] = {}
    for event in events:
        stats = per_model.setdefault(
            event["model_name"],
            {"total": 0, "PASS": 0, "FAIL": 0, "UNCERTAIN": 0,
             "latencies": [], "confidences": [], "gt_checked": 0, "gt_correct": 0},
        )
        stats["total"] += 1
        stats[event["decision"]] = stats.get(event["decision"], 0) + 1
        stats["latencies"].append(event["latency_ms"])
        stats["confidences"].append(event["confidence"])
        if event.get("ground_truth"):
            stats["gt_checked"] += 1
            if event["ground_truth"] == event["decision"]:
                stats["gt_correct"] += 1
    for stats in per_model.values():
        stats["mean_latency_ms"]  = statistics.mean(stats["latencies"]) if stats["latencies"] else 0.0
        stats["mean_confidence"]  = statistics.mean(stats["confidences"]) if stats["confidences"] else 0.0
        stats["running_accuracy"] = (
            stats["gt_correct"] / stats["gt_checked"] if stats["gt_checked"] else None
        )
        del stats["latencies"]
        del stats["confidences"]
    return {"total_events": len(events), "per_model": per_model}


def write_session_logs(
    session_dir: Path, session_name: str, session_payload: dict,
) -> tuple[Path, Path]:
    session_dir.mkdir(parents=True, exist_ok=True)
    json_path = session_dir / f"{session_name}.json"
    md_path   = session_dir / f"{session_name}.md"
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(session_payload, fh, indent=2)
    lines = [
        f"# {session_name}", "",
        f"- Started: {session_payload['started_at']}",
        f"- Ended:   {session_payload['ended_at']}",
        f"- Source:  {session_payload['source']}",
        f"- Total events: {session_payload['summary']['total_events']}", "",
        "## Per-model summary", "",
        "| Model | Total | PASS | FAIL | UNCERTAIN | Mean lat (ms) | Mean conf | GT checked | GT correct | Accuracy |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model_name, stats in session_payload["summary"]["per_model"].items():
        acc = f"{stats['running_accuracy']:.3f}" if stats["running_accuracy"] is not None else "-"
        lines.append(
            f"| {model_name} | {stats['total']} | {stats['PASS']} | {stats['FAIL']} | "
            f"{stats['UNCERTAIN']} | {stats['mean_latency_ms']:.1f} | {stats['mean_confidence']:.3f} | "
            f"{stats['gt_checked']} | {stats['gt_correct']} | {acc} |"
        )
    if session_payload.get("captures"):
        lines += ["", "## Captures", ""]
        for cap in session_payload["captures"]:
            lines.append(f"- frame={cap['frame_index']} reason={cap['reason']} path={cap['path']}")
    with md_path.open("w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return json_path, md_path


# ── CLI ──────────────────────────────────────────────────────────────────────

def resolve_output_dir(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def resolve_input_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    for candidate in [Path.cwd() / path, REPO_ROOT / path, STACK_ROOT / path]:
        if candidate.exists():
            return candidate
    return REPO_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Top-6 demo: runs top-5 models simultaneously plus an ensemble panel. "
            "Thresholds: PASS ≤ pass-threshold, FAIL ≥ fail-threshold (defect probability)."
        )
    )
    parser.add_argument(
        "--models", type=Path, nargs="+", default=DEFAULT_MODEL_PATHS,
        metavar="MODEL_PT",
        help="Checkpoint paths (default: top-5 from top10_models.md). Supply 1-N paths to override.",
    )
    parser.add_argument(
        "--model-names", type=str, nargs="+", default=None,
        metavar="NAME",
        help="Display names for each model (default: see DEFAULT_DISPLAY_NAMES).",
    )
    parser.add_argument("--input", type=Path, default=None, help="Image file, folder, or video.")
    parser.add_argument("--stream-url",     type=str,   default=env_or_none("DEMO_STREAM_URL"))
    parser.add_argument("--auth-mode",      type=str,   default="auto",
                        choices=["auto", "basic", "form", "none"])
    parser.add_argument("--auth-user",      type=str,   default=env_or_none("DEMO_AUTH_USER"))
    parser.add_argument("--auth-password",  type=str,   default=env_or_none("DEMO_AUTH_PASSWORD"))
    parser.add_argument("--login-url",      type=str,   default=env_or_none("DEMO_LOGIN_URL"))
    parser.add_argument("--frame-stride",   type=int,   default=1)
    parser.add_argument("--pass-threshold", type=float, default=DEFECT_THRESHOLD_PASS,
                        help="defect_prob ≤ this  →  PASS  (default 0.40)")
    parser.add_argument("--fail-threshold", type=float, default=DEFECT_THRESHOLD_FAIL,
                        help="defect_prob ≥ this  →  FAIL  (default 0.70)")
    parser.add_argument("--show",    dest="show", action="store_true")
    parser.add_argument("--no-show", dest="show", action="store_false")
    parser.add_argument("--display-width",  type=int,  default=1280)
    parser.add_argument("--display-height", type=int,  default=1024)
    parser.add_argument("--capture-dir",  type=Path, default=STACK_ROOT / "run_capture")
    parser.add_argument("--session-dir",  type=Path, default=DEFAULT_DEMO_DIR)
    parser.add_argument("--insecure",  action="store_true")
    parser.add_argument("--timeout",   type=float, default=10.0)
    parser.add_argument("--debug-bytes", type=int, default=4096)
    parser.set_defaults(show=True)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.stream_url and not args.input:
        raise SystemExit("[ERROR] Provide --stream-url or --input.")
    if args.stream_url and args.input:
        raise SystemExit("[ERROR] Use only one of --stream-url or --input.")
    if args.frame_stride < 1:
        raise SystemExit("[ERROR] --frame-stride must be >= 1.")
    if not (0.0 <= args.pass_threshold < args.fail_threshold <= 1.0):
        raise SystemExit("[ERROR] Need 0 ≤ pass-threshold < fail-threshold ≤ 1.")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    validate_args(args)

    args.capture_dir = resolve_output_dir(args.capture_dir)
    args.session_dir = resolve_output_dir(args.session_dir)

    # Resolve and load models
    model_paths = [
        p if p.is_absolute() else resolve_input_path(p)
        for p in args.models
    ]
    display_names = (
        args.model_names
        if args.model_names and len(args.model_names) == len(model_paths)
        else DEFAULT_DISPLAY_NAMES[:len(model_paths)]
    )

    print(f"[INFO] Loading {len(model_paths)} models on {DEVICE}...")
    try:
        demo_models = [
            DemoModel(
                path, name,
                pass_threshold=args.pass_threshold,
                fail_threshold=args.fail_threshold,
            )
            for path, name in zip(model_paths, display_names)
        ]
    except Exception as exc:
        raise SystemExit(f"[ERROR] Failed to load models: {exc}") from exc

    for m in demo_models:
        print(f"  [{m.model_type:10s}] {m.display_name}  →  {m.path}")

    # Frame source
    if args.input:
        input_path = resolve_input_path(args.input)
        if not input_path.exists():
            raise SystemExit(f"[ERROR] Input not found: {input_path}")
        frame_source  = iter_input_frames(input_path)
        source_label  = str(input_path)
        startup_lines = [
            f"models={', '.join(m.display_name for m in demo_models)}",
            f"input={input_path.name}",
            f"thresholds: PASS≤{args.pass_threshold:.2f}  FAIL≥{args.fail_threshold:.2f}",
        ]
    else:
        assert args.stream_url is not None
        stream_url, auth = resolve_basic_auth(
            args.stream_url, args.auth_mode, args.auth_user, args.auth_password
        )
        if args.insecure:
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        startup_lines = [
            f"models={', '.join(m.display_name for m in demo_models)}",
            f"stream={stream_url}",
            f"thresholds: PASS≤{args.pass_threshold:.2f}  FAIL≥{args.fail_threshold:.2f}",
        ]
        if auth is not None:
            startup_lines.append(f"user={auth[0]}")
        source_label = stream_url
        print(f"[INFO] connecting to stream: {stream_url}")
        frame_source = iter_mjpeg_frames(
            stream_url=stream_url, auth_mode=args.auth_mode, auth=auth,
            login_url=args.login_url, verify_tls=not args.insecure,
            timeout=args.timeout, debug_bytes=args.debug_bytes,
        )

    session_name       = f"demo_top6_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    session_started_at = datetime.now().isoformat()
    predictions_log:   list[dict] = []
    captures:          list[dict] = []
    ensemble_counts    = {"PASS": 0, "FAIL": 0, "UNCERTAIN": 0}
    pending_ground_truth = False
    last_ground_truth: str | None = None
    paused            = False
    window_name       = "demo-live-test-top6"
    last_predictions: list[ModelPrediction] = []
    last_ensemble:    ModelPrediction | None = None
    last_frame_index  = 0
    last_fps          = 0.0
    last_infer_time   = perf_counter()

    if args.show:
        print(f"[INFO] opening display window: {window_name}")
        prepare_display_window(window_name, args.display_width, args.display_height)

    try:
        for packet in frame_source:
            last_frame_index = packet.frame_index
            frame_size_text  = f"{packet.frame.shape[1]}x{packet.frame.shape[0]}"

            now     = perf_counter()
            elapsed = max(now - last_infer_time, 1e-9)
            last_fps       = 1.0 / elapsed
            last_infer_time = now

            if not paused and packet.frame_index % args.frame_stride == 0:
                frame_preds: list[ModelPrediction] = []
                for model in demo_models:
                    pred = model.predict(packet.frame)
                    frame_preds.append(pred)
                    predictions_log.append({
                        "timestamp":    datetime.now().isoformat(),
                        "frame_index":  packet.frame_index,
                        "source_label": packet.source_label,
                        "model_name":   pred.model_name,
                        "model_type":   pred.model_type,
                        "class_name":   pred.class_name,
                        "decision":     pred.decision,
                        "confidence":   pred.confidence,
                        "defect_prob":  pred.defect_prob,
                        "latency_ms":   pred.latency_ms,
                        "probabilities": pred.probabilities,
                        "ground_truth": None,
                        "capture_path": None,
                    })
                    print(
                        f"frame={packet.frame_index}\tmodel={pred.model_name}\t"
                        f"decision={pred.decision}\tdefect_prob={pred.defect_prob:.3f}\t"
                        f"latency_ms={pred.latency_ms:.1f}",
                        flush=True,
                    )

                ensemble = compute_ensemble(frame_preds, args.pass_threshold, args.fail_threshold)
                ensemble_counts[ensemble.decision] = ensemble_counts.get(ensemble.decision, 0) + 1
                print(
                    f"frame={packet.frame_index}\tENSEMBLE\t"
                    f"verdict={ensemble.decision}\tmean_defect={ensemble.defect_prob:.3f}",
                    flush=True,
                )
                predictions_log.append({
                    "timestamp":    datetime.now().isoformat(),
                    "frame_index":  packet.frame_index,
                    "source_label": packet.source_label,
                    "model_name":   ensemble.model_name,
                    "model_type":   "ensemble",
                    "class_name":   ensemble.class_name,
                    "decision":     ensemble.decision,
                    "confidence":   ensemble.confidence,
                    "defect_prob":  ensemble.defect_prob,
                    "latency_ms":   0.0,
                    "probabilities": ensemble.probabilities,
                    "ground_truth": None,
                    "capture_path": None,
                })

                last_predictions = frame_preds
                last_ensemble    = ensemble

            annotated = draw_overlay(
                frame=packet.frame,
                model_predictions=last_predictions,
                ensemble=last_ensemble,
                frame_index=packet.frame_index,
                fps=last_fps,
                paused=paused,
                source_label=packet.source_label,
                frame_size_text=frame_size_text,
                startup_lines=startup_lines,
                pending_ground_truth=pending_ground_truth,
                last_ground_truth=last_ground_truth,
                ensemble_counts=ensemble_counts,
                pass_threshold=args.pass_threshold,
                fail_threshold=args.fail_threshold,
            )

            if args.show:
                cv2.imshow(window_name, fit_for_display(annotated, args.display_width, args.display_height))
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("p"):
                    paused = not paused
                elif key == ord(" "):
                    out = save_capture(args.capture_dir, annotated, packet.frame_index, "manual")
                    captures.append({"frame_index": packet.frame_index, "reason": "manual", "path": str(out)})
                    for event in reversed(predictions_log):
                        if event["frame_index"] != packet.frame_index:
                            break
                        if event["capture_path"] is None:
                            event["capture_path"] = str(out)
                    print(f"[INFO] manual capture saved: {out}")
                elif key == ord("g"):
                    pending_ground_truth = True
                elif pending_ground_truth and key in GT_HOTKEY_MAP:
                    last_ground_truth    = GT_HOTKEY_MAP[key]
                    pending_ground_truth = False
                    updated = 0
                    for event in reversed(predictions_log):
                        if event["frame_index"] != packet.frame_index:
                            break
                        event["ground_truth"] = last_ground_truth
                        updated += 1
                    print(f"[INFO] GT set frame {packet.frame_index}: {last_ground_truth} ({updated} rows)")

    except KeyboardInterrupt:
        pass
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "unknown"
        raise SystemExit(f"[ERROR] HTTP {status} on stream.") from exc
    except requests.RequestException as exc:
        raise SystemExit(f"[ERROR] Request error: {exc}") from exc
    except RuntimeError as exc:
        raise SystemExit(f"[ERROR] {exc}") from exc
    finally:
        if args.show:
            cv2.destroyAllWindows()

    session_payload = {
        "session_name":    session_name,
        "started_at":      session_started_at,
        "ended_at":        datetime.now().isoformat(),
        "source":          source_label,
        "last_frame_index": last_frame_index,
        "thresholds":      {"pass": args.pass_threshold, "fail": args.fail_threshold},
        "models": [
            {"name": m.display_name, "path": str(m.path), "type": m.model_type,
             "class_names": m.class_names}
            for m in demo_models
        ],
        "captures": captures,
        "events":   predictions_log,
        "summary":  session_summary(predictions_log),
    }
    json_path, md_path = write_session_logs(args.session_dir, session_name, session_payload)

    print("\n=== Demo Top-6 Session Summary ===")
    print(f"source={source_label}")
    print(f"ensemble P/F/U  {ensemble_counts['PASS']}/{ensemble_counts['FAIL']}/{ensemble_counts['UNCERTAIN']}")
    for model_name, stats in session_payload["summary"]["per_model"].items():
        acc = f"{stats['running_accuracy']:.3f}" if stats["running_accuracy"] is not None else "-"
        print(
            f"  {model_name}: total={stats['total']} P={stats['PASS']} F={stats['FAIL']} "
            f"U={stats['UNCERTAIN']} lat={stats['mean_latency_ms']:.1f}ms acc={acc}"
        )
    print(f"session_json={json_path}")
    print(f"session_md={md_path}")


if __name__ == "__main__":
    main()
