#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import re
from time import perf_counter
from typing import Iterator
from urllib.parse import unquote, urljoin, urlsplit, urlunsplit

import cv2
import numpy as np
import requests
import urllib3
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PASS_FAIL_MAP = {
    "defective": "FAIL",
    "non_defective": "PASS",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run YOLO classification inference on an authenticated MJPEG/web stream and print PASS/FAIL decisions."
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--stream-url", type=str, required=True, help="HTTP/HTTPS MJPEG stream URL")
    parser.add_argument(
        "--auth-mode",
        type=str,
        default="auto",
        choices=["auto", "basic", "form", "none"],
        help="Authentication mode. 'auto' tries basic auth first, then form login if a login page is returned.",
    )
    parser.add_argument("--auth-user", type=str, default=None, help="Authentication username override")
    parser.add_argument("--auth-password", type=str, default=None, help="Authentication password override")
    parser.add_argument("--login-url", type=str, default=None, help="Optional explicit login form URL")
    parser.add_argument("--frame-stride", type=int, default=1, help="Run inference every N decoded frames")
    parser.add_argument("--show", action="store_true", help="Display the stream with overlayed predictions")
    parser.add_argument(
        "--display-width",
        type=int,
        default=1280,
        help="Maximum display width when --show is enabled.",
    )
    parser.add_argument(
        "--display-height",
        type=int,
        default=720,
        help="Maximum display height when --show is enabled.",
    )
    parser.add_argument(
        "--capture-dir",
        type=Path,
        default=PROJECT_ROOT / "run_capture",
        help="Directory for annotated capture images.",
    )
    parser.add_argument("--save-every", type=int, default=30, help="Save one annotated frame every N decoded frames")
    parser.add_argument("--insecure", action="store_true", help="Disable TLS certificate verification")
    parser.add_argument("--timeout", type=float, default=10.0, help="Initial connection timeout in seconds")
    parser.add_argument(
        "--debug-bytes",
        type=int,
        default=4096,
        help="Maximum number of initial response bytes to inspect when no JPEG frames are found.",
    )
    return parser.parse_args()


def sanitize_stream_url(url: str) -> str:
    parts = urlsplit(url)
    host = parts.hostname or ""
    if parts.port:
        host = f"{host}:{parts.port}"
    return urlunsplit((parts.scheme, host, parts.path, parts.query, parts.fragment))


def resolve_basic_auth(
    stream_url: str, auth_mode: str, auth_user: str | None, auth_password: str | None
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
                "Selected auth mode requires both --auth-user and --auth-password, or credentials in the URL."
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


def iter_mjpeg_frames(
    stream_url: str,
    auth_mode: str,
    auth: tuple[str, str] | None,
    login_url: str | None,
    verify_tls: bool,
    timeout: float,
    debug_bytes: int,
) -> Iterator:
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
                        yield frame

            if not saw_frame:
                preview = bytes(buffer[:debug_bytes]).decode("utf-8", errors="replace")
                raise RuntimeError(
                    "No JPEG frames were decoded from the response stream. "
                    f"HTTP content-type was {content_type!r}. "
                    f"Initial response preview: {preview!r}"
                )


def annotate_frame(frame, pred_class: str | None, decision: str | None, confidence: float | None, frame_index: int):
    label = decision or "ANALYZING"
    detail = f"class={pred_class or 'pending'}"
    if confidence is not None:
        detail += f" conf={confidence:.4f}"

    if label == "FAIL":
        color = (0, 0, 255)
    elif label == "PASS":
        color = (0, 180, 0)
    else:
        color = (0, 165, 255)

    annotated = frame.copy()
    overlay = annotated.copy()
    cv2.rectangle(overlay, (12, 12), (520, 122), color, -1)
    cv2.addWeighted(overlay, 0.22, annotated, 0.78, 0, annotated)

    cv2.putText(annotated, label, (28, 58), cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 4, cv2.LINE_AA)
    cv2.putText(annotated, detail, (28, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(
        annotated,
        f"frame={frame_index}",
        (28, 116),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return annotated


def save_capture(capture_dir: Path, annotated_frame, decision: str, frame_index: int) -> Path:
    capture_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{decision.lower()}_{timestamp}_PDE4444_f{frame_index:06d}.jpg"
    output_path = capture_dir / filename
    cv2.imwrite(str(output_path), annotated_frame)
    return output_path


def resolve_output_dir(path: Path) -> Path:
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def prepare_display_window(window_name: str, width: int, height: int) -> None:
    flags = cv2.WINDOW_NORMAL
    if hasattr(cv2, "WINDOW_KEEPRATIO"):
        flags |= cv2.WINDOW_KEEPRATIO
    cv2.namedWindow(window_name, flags)
    cv2.resizeWindow(window_name, width, height)


def fit_for_display(frame, max_width: int, max_height: int):
    height, width = frame.shape[:2]
    if width <= max_width and height <= max_height:
        return frame

    scale = min(max_width / width, max_height / height)
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)


def main() -> None:
    args = parse_args()
    if args.frame_stride < 1:
        raise ValueError("--frame-stride must be >= 1")
    if args.save_every < 1:
        raise ValueError("--save-every must be >= 1")
    if args.display_width < 1 or args.display_height < 1:
        raise ValueError("--display-width and --display-height must be >= 1")

    stream_url, auth = resolve_basic_auth(args.stream_url, args.auth_mode, args.auth_user, args.auth_password)
    args.capture_dir = resolve_output_dir(args.capture_dir)
    model = YOLO(str(args.model))

    if args.insecure:
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    print(f"[INFO] connecting to stream: {stream_url}")
    if auth is not None:
        print(f"[INFO] using HTTP basic auth for user: {auth[0]}")

    window_name = "zeroq-stream"
    if args.show:
        prepare_display_window(window_name, args.display_width, args.display_height)

    last_pred_class: str | None = None
    last_decision: str | None = None
    last_confidence: float | None = None

    try:
        for frame_index, frame in enumerate(
            iter_mjpeg_frames(
                stream_url,
                auth_mode=args.auth_mode,
                auth=auth,
                login_url=args.login_url,
                verify_tls=not args.insecure,
                timeout=args.timeout,
                debug_bytes=args.debug_bytes,
            ),
            start=1,
        ):
            if frame_index % args.frame_stride == 0:
                start_time = perf_counter()
                result = model.predict(source=frame, verbose=False)[0]
                process_ms = (perf_counter() - start_time) * 1000.0
                top1_index = int(result.probs.top1)
                last_pred_class = result.names[top1_index]
                last_confidence = float(result.probs.top1conf)
                last_decision = PASS_FAIL_MAP.get(last_pred_class, last_pred_class.upper())

                print(f"frame={frame_index}\tprocess_ms={process_ms:.1f}", flush=True)

            annotated = annotate_frame(frame, last_pred_class, last_decision, last_confidence, frame_index)

            if last_decision is not None and frame_index % args.save_every == 0:
                output_path = save_capture(args.capture_dir, annotated, last_decision, frame_index)
                print(f"[INFO] saved capture: {output_path}")

            if args.show:
                cv2.imshow(window_name, fit_for_display(annotated, args.display_width, args.display_height))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
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


if __name__ == "__main__":
    main()
