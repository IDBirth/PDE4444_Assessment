#!/usr/bin/env python3
# Example usage:
# python /home/ubu/Desktop/Assessment/defect_classification_stack/demo_live_test.py \
#   --model /home/ubu/Desktop/Assessment/runs3/iter1_yolo_320/train/weights/best.pt \
#   --stream-url "https://api.baslarlabs.ae/video_feed" \
#   --auth-user 'zeroq' \
#   --auth-password '#@45459xQw'
#
# Offline fallback:
# python /home/ubu/Desktop/Assessment/defect_classification_stack/demo_live_test.py \
#   --model /home/ubu/Desktop/Assessment/runs3/iter1_yolo_320/train/weights/best.pt \
#   --input /home/ubu/Desktop/Assessment/path/to/images_or_video
from __future__ import annotations

import argparse
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

REPO_ROOT = Path(__file__).resolve().parent.parent
STACK_ROOT = Path(__file__).resolve().parent
DEFAULT_DEMO_DIR = STACK_ROOT / "demo_sessions"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
GT_HOTKEY_MAP = {
    ord("1"): "PASS",
    ord("2"): "FAIL",
}

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)


def env_or_none(name: str) -> str | None:
    value = os.getenv(name)
    return value if value else None


def map_label_to_decision(label: str) -> str:
    normalized = label.strip().lower()
    if normalized in {"non_defective", "non-defective", "non_defect", "pass"}:
        return "PASS"
    if normalized in {"defective", "defect", "fail"}:
        return "FAIL"
    if "non" in normalized and "def" in normalized:
        return "PASS"
    if "def" in normalized:
        return "FAIL"
    return normalized.upper()


def sanitize_stream_url(url: str) -> str:
    parts = urlsplit(url)
    host = parts.hostname or ""
    if parts.port:
        host = f"{host}:{parts.port}"
    return urlunsplit((parts.scheme, host, parts.path, parts.query, parts.fragment))


def resolve_basic_auth(
    stream_url: str,
    auth_mode: str,
    auth_user: str | None,
    auth_password: str | None,
) -> tuple[str, tuple[str, str] | None]:
    parts = urlsplit(stream_url)
    sanitized_url = sanitize_stream_url(stream_url)

    if auth_mode == "none":
        return sanitized_url, None

    user = auth_user
    password = auth_password

    if auth_mode in {"auto", "basic", "form"}:
        if user is None and parts.username is not None:
            user = unquote(parts.username)
        if password is None and parts.password is not None:
            password = unquote(parts.password)

    if auth_mode in {"basic", "form"}:
        if user is None or password is None:
            raise ValueError(
                "Selected auth mode requires --auth-user and --auth-password, "
                "credentials in the URL, or matching environment variables."
            )
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
    action_url = urljoin(base_url, action_match.group(1)) if action_match else base_url

    payload: dict[str, str] = {}
    hidden_input_pattern = re.compile(
        r"<input[^>]*type=[\"']hidden[\"'][^>]*name=[\"']([^\"']+)[\"'][^>]*value=[\"']([^\"']*)[\"']",
        flags=re.IGNORECASE,
    )
    for match in hidden_input_pattern.finditer(html):
        payload[match.group(1)] = match.group(2)

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

    print(f"[INFO] detected HTML login form, submitting credentials to: {login_url}")
    response = session.post(
        login_url,
        data=payload,
        timeout=timeout,
        verify=verify_tls,
        headers={"Referer": login_page_response.url},
        allow_redirects=True,
    )
    response.raise_for_status()
    print(f"[INFO] login POST HTTP {response.status_code}, final URL={response.url}")


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
            session=session,
            login_page_response=response,
            auth=auth,
            login_url_override=login_url,
            verify_tls=verify_tls,
            timeout=timeout,
        )
        response.close()
        response = session.get(stream_url, stream=True, timeout=(timeout, None), verify=verify_tls)
        response.raise_for_status()

    return response


@dataclass
class FramePacket:
    frame_index: int
    frame: np.ndarray
    source_label: str


@dataclass
class ModelPrediction:
    model_name: str
    model_type: str
    class_name: str
    decision: str
    confidence: float
    latency_ms: float
    probabilities: dict[str, float]
    thresholded: bool


class DemoModel:
    def __init__(self, model_path: Path, display_name: str | None = None) -> None:
        self.path = model_path
        self.display_name = display_name or model_path.stem
        self.model_type = detect_model_type(model_path)
        self.class_names = infer_class_names_for_model(model_path, self.model_type)

        if self.model_type == "yolo":
            self._yolo = YOLO(str(model_path))
            self._mobilenet = None
            self._transform = None
        elif self.model_type == "mobilenet":
            self._yolo = None
            activation_name = detect_activation_from_path(model_path)
            class_names = infer_class_names_from_report(model_path)
            self.class_names = class_names
            self._mobilenet = build_mobilenet_model(activation_name, model_path)
            self._transform = build_mobilenet_transform()
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def predict(self, frame: np.ndarray, confidence_threshold: float) -> ModelPrediction:
        start = perf_counter()

        if self.model_type == "yolo":
            result = self._yolo.predict(source=frame, verbose=False)[0]
            probs_tensor = result.probs.data.detach().cpu().float()
            top_index = int(result.probs.top1)
            class_name = str(result.names[top_index])
            confidence = float(probs_tensor[top_index].item())
            probabilities = {
                str(result.names[i]): float(prob.item())
                for i, prob in enumerate(probs_tensor)
            }
        else:
            assert self._mobilenet is not None
            assert self._transform is not None
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tensor = self._transform(rgb).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                logit = self._mobilenet(tensor).squeeze(0)
                positive_prob = float(torch.sigmoid(logit).item())

            negative_label, positive_label = self.class_names
            probabilities = {
                negative_label: 1.0 - positive_prob,
                positive_label: positive_prob,
            }
            class_name = positive_label if positive_prob >= 0.5 else negative_label
            confidence = max(probabilities.values())

        latency_ms = (perf_counter() - start) * 1000.0
        decision = map_label_to_decision(class_name)
        thresholded = confidence < confidence_threshold
        if thresholded:
            decision = "UNCERTAIN"

        return ModelPrediction(
            model_name=self.display_name,
            model_type=self.model_type,
            class_name=class_name,
            decision=decision,
            confidence=confidence,
            latency_ms=latency_ms,
            probabilities=probabilities,
            thresholded=thresholded,
        )


def build_mobilenet_transform() -> transforms.Compose:
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def activation_module(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU()
    if name == "elu":
        return nn.ELU()
    if name == "gelu":
        return nn.GELU()
    if name == "selu":
        return nn.SELU()
    if name == "leaky_relu":
        return nn.LeakyReLU()
    raise ValueError(f"Unsupported activation: {name}")


def build_mobilenet_model(activation_name: str, checkpoint_path: Path) -> nn.Module:
    backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    backbone.classifier = nn.Sequential(
        nn.Linear(backbone.last_channel, 256),
        nn.BatchNorm1d(256),
        activation_module(activation_name),
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
        "Could not infer MobileNet activation from checkpoint path. "
        f"Expected one of relu/elu/gelu/selu/leaky_relu in: {path}"
    )


def infer_class_names_from_report(path: Path) -> list[str]:
    report_path = path.parent / "classification_report.csv"
    if not report_path.exists():
        return ["non_defect", "defect"]

    rows: list[str] = []
    with report_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
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

    if len(rows) == 2:
        return rows
    return ["non_defect", "defect"]


def infer_class_names_for_model(path: Path, model_type: str) -> list[str]:
    if model_type == "mobilenet":
        return infer_class_names_from_report(path)
    return ["defective", "non_defective"]


def detect_model_type(path: Path) -> str:
    try:
        model = YOLO(str(path))
        if getattr(model, "task", None) == "classify":
            return "yolo"
    except Exception:
        pass

    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError(f"Could not determine model type for {path}: {exc}") from exc

    if isinstance(state, dict) and any(str(key).startswith("features.") for key in state.keys()):
        return "mobilenet"

    raise ValueError(
        "Unsupported checkpoint format. Expected a YOLO classification model or "
        "a MobileNetV2 state_dict checkpoint."
    )


def resolve_output_dir(path: Path) -> Path:
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def resolve_input_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    candidates = [Path.cwd() / path, REPO_ROOT / path, STACK_ROOT / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return REPO_ROOT / path


def resolve_model_path(path: Path) -> Path:
    resolved = resolve_input_path(path)
    if resolved.exists():
        return resolved

    available = list_available_models(limit=20)
    suggestions = "\n".join(f"  - {candidate}" for candidate in available) or "  <none found>"
    raise SystemExit(
        f"[ERROR] Model not found: {path}\n"
        f"Resolved path: {resolved}\n"
        "Available model candidates:\n"
        f"{suggestions}"
    )


def list_available_models(limit: int = 20) -> list[str]:
    search_roots = [
        REPO_ROOT / "defect_classification_stack" / "models",
        REPO_ROOT / "defect_classification_stack" / "runs",
        REPO_ROOT / "runs2",
        REPO_ROOT / "runs3",
        REPO_ROOT,
        REPO_ROOT / "zeroq_cup_classification_scaffold",
    ]
    found: list[str] = []
    seen: set[str] = set()
    for root in search_roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.pt")):
            path_str = str(path)
            if "__MACOSX" in path_str or ".venv" in path_str:
                continue
            rel = os.path.relpath(path, REPO_ROOT)
            if rel not in seen:
                seen.add(rel)
                found.append(rel)
            if len(found) >= limit:
                return found
    return found


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
            session=session,
            stream_url=stream_url,
            auth_mode=auth_mode,
            auth=auth,
            login_url=login_url,
            verify_tls=verify_tls,
            timeout=timeout,
        ) as response:
            response.raise_for_status()
            content_type = response.headers.get("content-type", "<missing>")
            print(f"[INFO] stream HTTP {response.status_code}, content-type={content_type}")
            buffer = bytearray()
            saw_frame = False
            frame_index = 0
            for chunk in response.iter_content(chunk_size=8192):
                if not chunk:
                    continue
                buffer.extend(chunk)

                while True:
                    start = buffer.find(b"\xff\xd8")
                    end = buffer.find(b"\xff\xd9", start + 2 if start != -1 else 0)
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
                    "No JPEG frames were decoded from the response stream. "
                    f"HTTP content-type was {content_type!r}. "
                    f"Initial response preview: {preview!r}"
                )


def iter_input_frames(path: Path) -> Iterator[FramePacket]:
    if path.is_dir():
        image_paths = sorted([p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS])
        if not image_paths:
            raise FileNotFoundError(f"No images found in directory: {path}")
        for frame_index, image_path in enumerate(image_paths, start=1):
            frame = cv2.imread(str(image_path))
            if frame is None:
                continue
            yield FramePacket(frame_index=frame_index, frame=frame, source_label=image_path.name)
        return

    if path.suffix.lower() in IMAGE_EXTENSIONS:
        frame = cv2.imread(str(path))
        if frame is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        yield FramePacket(frame_index=1, frame=frame, source_label=path.name)
        return

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")

    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_index += 1
            yield FramePacket(frame_index=frame_index, frame=frame, source_label=path.name)
    finally:
        capture.release()


def fit_for_display(frame: np.ndarray, max_width: int, max_height: int) -> np.ndarray:
    height, width = frame.shape[:2]
    if width <= max_width and height <= max_height:
        return frame
    scale = min(max_width / width, max_height / height)
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)


def prepare_display_window(window_name: str, width: int, height: int) -> None:
    flags = cv2.WINDOW_NORMAL
    if hasattr(cv2, "WINDOW_KEEPRATIO"):
        flags |= cv2.WINDOW_KEEPRATIO
    cv2.namedWindow(window_name, flags)
    cv2.resizeWindow(window_name, width, height)


def pick_two_probabilities(probabilities: dict[str, float]) -> list[tuple[str, float]]:
    items = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
    return items[:2]


def draw_probability_bar(
    frame: np.ndarray,
    x: int,
    y: int,
    width: int,
    height: int,
    fraction: float,
    fill_color: tuple[int, int, int],
) -> None:
    cv2.rectangle(frame, (x, y), (x + width, y + height), (80, 80, 80), 1)
    cv2.rectangle(frame, (x + 1, y + 1), (x + max(1, int((width - 2) * max(0.0, min(1.0, fraction)))), y + height - 1), fill_color, -1)


def save_capture(
    capture_dir: Path,
    annotated_frame: np.ndarray,
    frame_index: int,
    reason: str,
) -> Path:
    capture_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{reason}_{timestamp}_f{frame_index:06d}.jpg"
    output_path = capture_dir / filename
    cv2.imwrite(str(output_path), annotated_frame)
    return output_path


def session_summary(events: Sequence[dict]) -> dict:
    if not events:
        return {
            "total_events": 0,
            "per_model": {},
        }

    per_model: dict[str, dict] = {}
    for event in events:
        model_name = event["model_name"]
        stats = per_model.setdefault(
            model_name,
            {
                "total": 0,
                "PASS": 0,
                "FAIL": 0,
                "UNCERTAIN": 0,
                "latencies": [],
                "confidences": [],
                "gt_checked": 0,
                "gt_correct": 0,
            },
        )
        stats["total"] += 1
        stats[event["decision"]] = stats.get(event["decision"], 0) + 1
        stats["latencies"].append(event["latency_ms"])
        stats["confidences"].append(event["confidence"])
        if event.get("ground_truth"):
            stats["gt_checked"] += 1
            if event.get("ground_truth") == event["decision"]:
                stats["gt_correct"] += 1

    for stats in per_model.values():
        stats["mean_latency_ms"] = statistics.mean(stats["latencies"]) if stats["latencies"] else 0.0
        stats["mean_confidence"] = statistics.mean(stats["confidences"]) if stats["confidences"] else 0.0
        stats["running_accuracy"] = (
            stats["gt_correct"] / stats["gt_checked"] if stats["gt_checked"] else None
        )
        del stats["latencies"]
        del stats["confidences"]

    return {
        "total_events": len(events),
        "per_model": per_model,
    }


def write_session_logs(session_dir: Path, session_name: str, session_payload: dict) -> tuple[Path, Path]:
    session_dir.mkdir(parents=True, exist_ok=True)
    json_path = session_dir / f"{session_name}.json"
    md_path = session_dir / f"{session_name}.md"

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(session_payload, handle, indent=2)

    lines = [
        f"# {session_name}",
        "",
        f"- Started: {session_payload['started_at']}",
        f"- Ended: {session_payload['ended_at']}",
        f"- Source: {session_payload['source']}",
        f"- Total decision events: {session_payload['summary']['total_events']}",
        "",
        "## Per-model summary",
        "",
        "| Model | Total | PASS | FAIL | UNCERTAIN | Mean latency (ms) | Mean confidence | GT checked | GT correct | Running accuracy |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for model_name, stats in session_payload["summary"]["per_model"].items():
        accuracy_text = (
            f"{stats['running_accuracy']:.3f}" if stats["running_accuracy"] is not None else "-"
        )
        lines.append(
            f"| {model_name} | {stats['total']} | {stats['PASS']} | {stats['FAIL']} | "
            f"{stats['UNCERTAIN']} | {stats['mean_latency_ms']:.1f} | {stats['mean_confidence']:.3f} | "
            f"{stats['gt_checked']} | {stats['gt_correct']} | {accuracy_text} |"
        )

    if session_payload["captures"]:
        lines.extend(["", "## Captures", ""])
        for capture in session_payload["captures"]:
            lines.append(
                f"- frame={capture['frame_index']} reason={capture['reason']} path={capture['path']}"
            )

    with md_path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")

    return json_path, md_path


def draw_overlay(
    frame: np.ndarray,
    predictions: Sequence[ModelPrediction],
    frame_index: int,
    fps: float,
    paused: bool,
    source_label: str,
    frame_size_text: str,
    startup_lines: Sequence[str],
    pending_ground_truth: bool,
    last_ground_truth: str | None,
    running_counts: dict[str, int],
) -> np.ndarray:
    annotated = frame.copy()
    height, width = annotated.shape[:2]

    panel_width = min(720, width - 20)
    panel_height = min(250 + 115 * len(predictions), height - 20)
    overlay = annotated.copy()
    cv2.rectangle(overlay, (10, 10), (10 + panel_width, 10 + panel_height), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.72, annotated, 0.28, 0, annotated)

    y = 36
    cv2.putText(annotated, "DEMO LIVE TEST", (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (240, 240, 240), 2, cv2.LINE_AA)
    y += 28
    status = "PAUSED" if paused else "RUNNING"
    gt_text = "awaiting 1/2" if pending_ground_truth else (last_ground_truth or "-")
    top_line = (
        f"frame={frame_index} fps={fps:.1f} status={status} "
        f"counts(P/F/U)={running_counts['PASS']}/{running_counts['FAIL']}/{running_counts['UNCERTAIN']}"
    )
    cv2.putText(annotated, top_line, (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
    y += 22
    cv2.putText(
        annotated,
        f"source={source_label} size={frame_size_text} gt={gt_text}",
        (24, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    y += 28

    for line in startup_lines[:4]:
        cv2.putText(annotated, line, (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 210, 255), 1, cv2.LINE_AA)
        y += 18
    if startup_lines:
        y += 8

    for prediction in predictions:
        if prediction.decision == "FAIL":
            color = (0, 0, 255)
        elif prediction.decision == "PASS":
            color = (0, 180, 0)
        else:
            color = (0, 165, 255)

        cv2.putText(
            annotated,
            f"{prediction.model_name} [{prediction.model_type}]",
            (24, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y += 24

        cv2.putText(
            annotated,
            f"decision={prediction.decision} class={prediction.class_name} conf={prediction.confidence:.3f} latency={prediction.latency_ms:.1f}ms",
            (24, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            color,
            1,
            cv2.LINE_AA,
        )
        y += 20

        probs = pick_two_probabilities(prediction.probabilities)
        for idx, (label, prob) in enumerate(probs):
            bar_y = y + idx * 18
            draw_probability_bar(annotated, 24, bar_y - 10, 180, 10, prob, color if idx == 0 else (160, 160, 160))
            cv2.putText(
                annotated,
                f"{label}={prob:.3f}",
                (214, bar_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (220, 220, 220),
                1,
                cv2.LINE_AA,
            )
        y += 48

    help_text = "SPACE=capture  g+1/2=GT PASS/FAIL  p=pause  q=quit"
    cv2.putText(
        annotated,
        help_text,
        (24, min(height - 18, 10 + panel_height - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )
    return annotated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Final demo runner for stream or offline PASS/FAIL inference with evidence logging."
    )
    parser.add_argument("--model", type=Path, required=True, help="Primary model checkpoint.")
    parser.add_argument("--second-model", type=Path, default=None, help="Optional comparison model checkpoint.")
    parser.add_argument("--input", type=Path, default=None, help="Image file, image folder, or video file.")
    parser.add_argument(
        "--stream-url",
        type=str,
        default=env_or_none("DEMO_STREAM_URL"),
        help="HTTP/HTTPS MJPEG stream URL. Can also be set via DEMO_STREAM_URL.",
    )
    parser.add_argument(
        "--auth-mode",
        type=str,
        default="auto",
        choices=["auto", "basic", "form", "none"],
        help="Authentication mode. Auto tries basic first, then form login if needed.",
    )
    parser.add_argument("--auth-user", type=str, default=env_or_none("DEMO_AUTH_USER"))
    parser.add_argument("--auth-password", type=str, default=env_or_none("DEMO_AUTH_PASSWORD"))
    parser.add_argument("--login-url", type=str, default=env_or_none("DEMO_LOGIN_URL"))
    parser.add_argument("--frame-stride", type=int, default=1, help="Run inference every N frames.")
    parser.add_argument("--confidence-threshold", type=float, default=0.60)
    parser.add_argument("--show", action="store_true", help="Display annotated output window.")
    parser.add_argument("--display-width", type=int, default=1280)
    parser.add_argument("--display-height", type=int, default=720)
    parser.add_argument("--capture-dir", type=Path, default=STACK_ROOT / "run_capture")
    parser.add_argument("--session-dir", type=Path, default=DEFAULT_DEMO_DIR)
    parser.add_argument("--save-every", type=int, default=30, help="Auto-save one annotated frame every N frames.")
    parser.add_argument("--insecure", action="store_true", help="Disable TLS verification.")
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--debug-bytes", type=int, default=4096)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.stream_url and not args.input:
        raise SystemExit("[ERROR] Provide either --stream-url or --input.")
    if args.stream_url and args.input:
        raise SystemExit("[ERROR] Use only one of --stream-url or --input.")
    if args.frame_stride < 1:
        raise SystemExit("[ERROR] --frame-stride must be >= 1.")
    if args.save_every < 1:
        raise SystemExit("[ERROR] --save-every must be >= 1.")
    if args.display_width < 1 or args.display_height < 1:
        raise SystemExit("[ERROR] --display-width and --display-height must be >= 1.")
    if not (0.0 <= args.confidence_threshold <= 1.0):
        raise SystemExit("[ERROR] --confidence-threshold must be between 0 and 1.")


def main() -> None:
    args = parse_args()
    validate_args(args)

    args.capture_dir = resolve_output_dir(args.capture_dir)
    args.session_dir = resolve_output_dir(args.session_dir)
    model_paths = [resolve_model_path(args.model)]
    if args.second_model is not None:
        model_paths.append(resolve_model_path(args.second_model))

    try:
        models_to_run = [DemoModel(path) for path in model_paths]
    except Exception as exc:
        available = "\n".join(f"  - {path}" for path in list_available_models(limit=20))
        raise SystemExit(f"[ERROR] Failed to load model: {exc}\nAvailable model candidates:\n{available}") from exc

    source_label = "offline"
    startup_lines = [
        f"models={', '.join(model.display_name for model in models_to_run)}",
        f"auth_mode={args.auth_mode}",
        f"threshold={args.confidence_threshold:.2f}",
    ]

    if args.input:
        input_path = resolve_input_path(args.input)
        if not input_path.exists():
            raise SystemExit(f"[ERROR] Input path not found: {input_path}")
        frame_source = iter_input_frames(input_path)
        source_label = str(input_path)
        startup_lines.append(f"input={input_path.name}")
    else:
        assert args.stream_url is not None
        stream_url, auth = resolve_basic_auth(args.stream_url, args.auth_mode, args.auth_user, args.auth_password)
        if args.insecure:
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        startup_lines.append(f"stream={stream_url}")
        if auth is not None:
            startup_lines.append(f"user={auth[0]}")
        print(f"[INFO] connecting to stream: {stream_url}")
        frame_source = iter_mjpeg_frames(
            stream_url=stream_url,
            auth_mode=args.auth_mode,
            auth=auth,
            login_url=args.login_url,
            verify_tls=not args.insecure,
            timeout=args.timeout,
            debug_bytes=args.debug_bytes,
        )

    session_name = f"demo_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    session_started_at = datetime.now().isoformat()
    predictions_log: list[dict] = []
    captures: list[dict] = []
    running_counts = {"PASS": 0, "FAIL": 0, "UNCERTAIN": 0}
    pending_ground_truth = False
    last_ground_truth: str | None = None
    paused = False
    window_name = "demo-live-test"
    last_predictions: list[ModelPrediction] = []
    last_frame_index = 0
    last_source = source_label
    last_fps = 0.0
    last_infer_time = perf_counter()

    if args.show:
        prepare_display_window(window_name, args.display_width, args.display_height)

    try:
        for packet in frame_source:
            last_frame_index = packet.frame_index
            last_source = packet.source_label
            frame_size_text = f"{packet.frame.shape[1]}x{packet.frame.shape[0]}"

            now = perf_counter()
            elapsed = max(now - last_infer_time, 1e-9)
            last_fps = 1.0 / elapsed
            last_infer_time = now

            if not paused and packet.frame_index % args.frame_stride == 0:
                frame_predictions: list[ModelPrediction] = []
                for model in models_to_run:
                    prediction = model.predict(packet.frame, args.confidence_threshold)
                    frame_predictions.append(prediction)
                    running_counts[prediction.decision] = running_counts.get(prediction.decision, 0) + 1
                    predictions_log.append(
                        {
                            "timestamp": datetime.now().isoformat(),
                            "frame_index": packet.frame_index,
                            "source_label": packet.source_label,
                            "model_name": prediction.model_name,
                            "model_type": prediction.model_type,
                            "class_name": prediction.class_name,
                            "decision": prediction.decision,
                            "confidence": prediction.confidence,
                            "latency_ms": prediction.latency_ms,
                            "probabilities": prediction.probabilities,
                            "ground_truth": None,
                            "capture_path": None,
                        }
                    )
                    print(
                        f"frame={packet.frame_index}\tmodel={prediction.model_name}\t"
                        f"decision={prediction.decision}\tclass={prediction.class_name}\t"
                        f"confidence={prediction.confidence:.3f}\tlatency_ms={prediction.latency_ms:.1f}",
                        flush=True,
                    )
                last_predictions = frame_predictions

            annotated = draw_overlay(
                frame=packet.frame,
                predictions=last_predictions,
                frame_index=packet.frame_index,
                fps=last_fps,
                paused=paused,
                source_label=packet.source_label,
                frame_size_text=frame_size_text,
                startup_lines=startup_lines,
                pending_ground_truth=pending_ground_truth,
                last_ground_truth=last_ground_truth,
                running_counts=running_counts,
            )

            if last_predictions and packet.frame_index % args.save_every == 0:
                output_path = save_capture(args.capture_dir, annotated, packet.frame_index, "auto")
                captures.append(
                    {
                        "frame_index": packet.frame_index,
                        "reason": "auto",
                        "path": str(output_path),
                    }
                )
                for event in reversed(predictions_log):
                    if event["frame_index"] != packet.frame_index:
                        break
                    event["capture_path"] = str(output_path)
                print(f"[INFO] saved capture: {output_path}")

            if args.show:
                cv2.imshow(window_name, fit_for_display(annotated, args.display_width, args.display_height))
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("p"):
                    paused = not paused
                elif key == ord(" "):
                    output_path = save_capture(args.capture_dir, annotated, packet.frame_index, "manual")
                    captures.append(
                        {
                            "frame_index": packet.frame_index,
                            "reason": "manual",
                            "path": str(output_path),
                        }
                    )
                    for event in reversed(predictions_log):
                        if event["frame_index"] != packet.frame_index:
                            break
                        if event["capture_path"] is None:
                            event["capture_path"] = str(output_path)
                    print(f"[INFO] manual capture saved: {output_path}")
                elif key == ord("g"):
                    pending_ground_truth = True
                elif pending_ground_truth and key in GT_HOTKEY_MAP:
                    last_ground_truth = GT_HOTKEY_MAP[key]
                    pending_ground_truth = False
                    updated = 0
                    for event in reversed(predictions_log):
                        if event["frame_index"] != packet.frame_index:
                            break
                        event["ground_truth"] = last_ground_truth
                        updated += 1
                    print(f"[INFO] ground truth set for frame {packet.frame_index}: {last_ground_truth} ({updated} model predictions)")
            elif packet.frame_index == 1 and args.input and packet.source_label.endswith(tuple(IMAGE_EXTENSIONS)):
                # For single-image / folder mode without display, avoid replaying the same frame forever.
                continue
    except KeyboardInterrupt:
        pass
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "unknown"
        raise SystemExit(f"[ERROR] HTTP error while opening stream: status={status}") from exc
    except requests.RequestException as exc:
        raise SystemExit(f"[ERROR] Request error while opening stream: {exc}") from exc
    except RuntimeError as exc:
        raise SystemExit(f"[ERROR] {exc}") from exc
    finally:
        if args.show:
            cv2.destroyAllWindows()

    session_payload = {
        "session_name": session_name,
        "started_at": session_started_at,
        "ended_at": datetime.now().isoformat(),
        "source": source_label,
        "last_frame_index": last_frame_index,
        "models": [
            {
                "name": model.display_name,
                "path": str(model.path),
                "type": model.model_type,
                "class_names": model.class_names,
            }
            for model in models_to_run
        ],
        "captures": captures,
        "events": predictions_log,
        "summary": session_summary(predictions_log),
    }
    json_path, md_path = write_session_logs(args.session_dir, session_name, session_payload)

    print("\n=== Demo Session Summary ===")
    print(f"source={source_label}")
    for model_name, stats in session_payload["summary"]["per_model"].items():
        accuracy_text = (
            f"{stats['running_accuracy']:.3f}" if stats["running_accuracy"] is not None else "-"
        )
        print(
            f"{model_name}: total={stats['total']} PASS={stats['PASS']} FAIL={stats['FAIL']} "
            f"UNCERTAIN={stats['UNCERTAIN']} mean_latency_ms={stats['mean_latency_ms']:.1f} "
            f"mean_confidence={stats['mean_confidence']:.3f} gt_checked={stats['gt_checked']} "
            f"gt_correct={stats['gt_correct']} running_accuracy={accuracy_text}"
        )
    print(f"session_json={json_path}")
    print(f"session_md={md_path}")


if __name__ == "__main__":
    main()
