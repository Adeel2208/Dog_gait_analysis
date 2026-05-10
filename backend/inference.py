"""
inference.py — Dog pose estimation using the fine-tuned best.pt model.

Single-model pipeline (exactly as the notebook):
  - best.pt detects dogs AND outputs 24 anatomically correct keypoints in one pass
  - No cropping, no second model needed
  - 24 keypoints match the StanfordExtra dog skeleton from keypoint_definitions.csv
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from ultralytics import YOLO
from backend.gait_analysis import GaitAnalyser, GaitResult, WINDOW_FRAMES

# ── 24-keypoint color map from keypoint_definitions.csv ───────────────────────
_CSV_PATH = Path(__file__).resolve().parent.parent / "keypoint_definitions.csv"

NUM_KEYPOINTS = 24

# Keypoint names for reference
KPT_NAMES = [
    "L Front Paw", "L Front Knee", "L Front Elbow",
    "L Rear Paw",  "L Rear Knee",  "L Rear Elbow",
    "R Front Paw", "R Front Knee", "R Front Elbow",
    "R Rear Paw",  "R Rear Knee",  "R Rear Elbow",
    "Tail Base",   "Tail End",
    "L Ear Base",  "R Ear Base",
    "Nose",        "Chin",
    "L Ear Tip",   "R Ear Tip",
    "L Eye",       "R Eye",
    "Withers",     "Throat",
]

def _build_color_map():
    """BGR colors per keypoint index, loaded from keypoint_definitions.csv."""
    if _CSV_PATH.exists():
        df = pd.read_csv(_CSV_PATH)
        colors = []
        for hex_color in df["Hex colour"].values:
            h = str(hex_color).strip()
            r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
            colors.append((b, g, r))   # OpenCV BGR
        return colors
    # Fallback: HSV wheel
    return [
        tuple(int(v) for v in cv2.cvtColor(
            np.uint8([[[int(i * 180 / NUM_KEYPOINTS), 255, 220]]]),
            cv2.COLOR_HSV2BGR)[0][0])
        for i in range(NUM_KEYPOINTS)
    ]

KPT_COLORS = _build_color_map()   # list of 24 BGR tuples

# Skeleton connections — pairs of keypoint indices to draw lines between
# Groups: front-left leg, front-right leg, rear-left leg, rear-right leg,
#         spine/body, head
SKELETON = [
    # Front-left leg:  paw → knee → elbow
    (0, 1), (1, 2),
    # Front-right leg: paw → knee → elbow
    (6, 7), (7, 8),
    # Rear-left leg:   paw → knee → elbow
    (3, 4), (4, 5),
    # Rear-right leg:  paw → knee → elbow
    (9, 10), (10, 11),
    # Body:  L-elbow ↔ R-elbow (shoulders), L-rear-elbow ↔ R-rear-elbow (hips)
    (2, 8),   # front shoulder line
    (5, 11),  # rear hip line
    # Spine: front-shoulder → withers → rear-hip
    (2, 22), (8, 22),    # elbows → withers
    (5, 22), (11, 22),   # rear elbows → withers (approximate spine)
    # Neck/head: withers → throat → nose
    (22, 23), (23, 16),
    # Tail: tail-base → tail-end
    (12, 13),
    # Ears: ear-base → ear-tip
    (14, 18), (15, 19),
    # Eyes to nose
    (20, 16), (21, 16),
]

# Line color = average of the two endpoint keypoint colors
def _skeleton_color(i, j):
    c1, c2 = KPT_COLORS[i], KPT_COLORS[j]
    return tuple(int((a + b) / 2) for a, b in zip(c1, c2))

SKELETON_COLORS = [_skeleton_color(i, j) for i, j in SKELETON]


# ── Drawing helpers ────────────────────────────────────────────────────────────

def draw_box(image: np.ndarray, xyxy: np.ndarray,
             score: float = None) -> np.ndarray:
    h, w = image.shape[:2]
    fs = max(0.45, min(0.75, 0.22 + 0.06 * min(h, w) / 100))
    lw = max(2, int(min(h, w) / 250))
    x1, y1, x2, y2 = xyxy[:4].astype(int).tolist()
    color = (0, 220, 0)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, lw, cv2.LINE_AA)
    label = "dog" + (f" {score:.2f}" if score is not None else "")
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, 2)
    cv2.rectangle(image, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
    cv2.putText(image, label, (x1 + 3, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), 2, cv2.LINE_AA)
    return image


def draw_pose(image: np.ndarray,
              kpts_xy: np.ndarray,
              kpts_conf: np.ndarray,
              kpt_conf_thresh: float = 0.30) -> tuple:
    """
    Draw skeleton lines + colored keypoint dots for one dog instance.
    kpts_xy   : (24, 2) absolute pixel coords
    kpts_conf : (24,)   confidence per keypoint
    Returns (annotated_image, n_visible_keypoints).
    """
    h, w   = image.shape[:2]
    radius = max(5, int(min(h, w) / 70))
    lw     = max(2, int(min(h, w) / 180))

    visible = kpts_conf > kpt_conf_thresh   # (24,) bool

    # ── Skeleton lines (drawn first, underneath dots) ──────────────────────
    for idx, (i, j) in enumerate(SKELETON):
        if visible[i] and visible[j]:
            pt1 = tuple(kpts_xy[i].astype(int).tolist())
            pt2 = tuple(kpts_xy[j].astype(int).tolist())
            cv2.line(image, pt1, pt2, SKELETON_COLORS[idx], lw, cv2.LINE_AA)

    # ── Keypoint dots ──────────────────────────────────────────────────────
    n_vis = 0
    for kid in range(NUM_KEYPOINTS):
        if visible[kid]:
            x, y  = int(kpts_xy[kid][0]), int(kpts_xy[kid][1])
            color = KPT_COLORS[kid]
            cv2.circle(image, (x, y), radius,     color, -1, cv2.LINE_AA)
            cv2.circle(image, (x, y), radius + 1, (0, 0, 0), 1, cv2.LINE_AA)
            n_vis += 1

    return image, n_vis


# ── Gait overlay ──────────────────────────────────────────────────────────────

# Colors for severity levels
_SEVERITY_COLOR = {
    "Normal":       (0,   220,  0),    # green
    "Mild":         (0,   200, 255),   # yellow
    "Moderate":     (0,   140, 255),   # orange
    "Severe":       (0,    0,  220),   # red
    "Analysing...": (180, 180, 180),   # grey
}

_LEG_POSITIONS = {
    # (x_frac, y_frac) of frame size for each leg indicator dot
    "LF": (0.20, 0.82),
    "RF": (0.35, 0.82),
    "LR": (0.55, 0.82),
    "RR": (0.70, 0.82),
}

_LEG_LABELS = {"LF": "L.Front", "RF": "R.Front", "LR": "L.Rear", "RR": "R.Rear"}


def draw_gait_overlay(frame: np.ndarray, result: GaitResult) -> np.ndarray:
    """
    Draw a gait analysis panel on the bottom of the frame:
      - Status banner (Normal / Mild / Moderate / Severe Limp + affected leg)
      - Four leg indicators with SI scores
      - Step height bars
      - Reason text (first reason)
    """
    h, w = frame.shape[:2]
    panel_h = int(h * 0.22)
    panel_y = h - panel_h

    # Semi-transparent dark panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, panel_y), (w, h), (10, 10, 10), -1)
    frame = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)

    sev_color = _SEVERITY_COLOR.get(result.severity, (180, 180, 180))
    fs_big    = max(0.55, min(0.9, w / 700))
    fs_small  = max(0.38, min(0.6, w / 900))
    lw        = max(1, int(w / 500))

    # ── Status banner ──────────────────────────────────────────────────────────
    banner_y = panel_y + int(panel_h * 0.30)
    cv2.putText(frame, result.label,
                (12, banner_y),
                cv2.FONT_HERSHEY_SIMPLEX, fs_big,
                sev_color, lw + 1, cv2.LINE_AA)

    # ── Reason line ────────────────────────────────────────────────────────────
    if result.reasons:
        reason_y = panel_y + int(panel_h * 0.58)
        cv2.putText(frame, result.reasons[0][:70],
                    (12, reason_y),
                    cv2.FONT_HERSHEY_SIMPLEX, fs_small,
                    (200, 200, 200), lw, cv2.LINE_AA)

    # ── Per-leg indicators ─────────────────────────────────────────────────────
    dot_r   = max(8, int(w / 60))
    label_y = panel_y + int(panel_h * 0.82)
    si_y    = panel_y + int(panel_h * 0.97)

    for leg, (xf, yf) in _LEG_POSITIONS.items():
        cx = int(w * xf)
        cy = panel_y + int(panel_h * yf)

        # Color: red if this is the affected leg, else green/grey
        if leg == result.affected_leg:
            dot_color = _SEVERITY_COLOR.get(result.severity, (0, 0, 220))
            ring_lw   = 3
        else:
            dot_color = (0, 180, 0)
            ring_lw   = 1

        cv2.circle(frame, (cx, cy), dot_r, dot_color, -1, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), dot_r + 2, (255, 255, 255), ring_lw, cv2.LINE_AA)

        # Leg label
        lbl = _LEG_LABELS[leg]
        (tw, _), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, fs_small, 1)
        cv2.putText(frame, lbl, (cx - tw // 2, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, fs_small,
                    (220, 220, 220), lw, cv2.LINE_AA)

        # SI score below label
        pair = "LF/RF" if leg in ("LF", "RF") else "LR/RR"
        si   = result.si_step.get(pair, 0.0)
        si_txt = f"SI:{si:.0f}%"
        (tw2, _), _ = cv2.getTextSize(si_txt, cv2.FONT_HERSHEY_SIMPLEX, fs_small * 0.85, 1)
        si_color = (0, 80, 255) if si > 15 else (100, 220, 100)
        cv2.putText(frame, si_txt, (cx - tw2 // 2, si_y),
                    cv2.FONT_HERSHEY_SIMPLEX, fs_small * 0.85,
                    si_color, lw, cv2.LINE_AA)

    # ── Step height mini-bars ──────────────────────────────────────────────────
    max_sh   = max(result.step_heights.values()) if result.step_heights else 1.0
    bar_w    = max(6, int(w / 80))
    bar_maxh = int(panel_h * 0.35)
    bar_base = panel_y + int(panel_h * 0.72)

    for leg, (xf, _) in _LEG_POSITIONS.items():
        cx  = int(w * xf) + dot_r + 6
        sh  = result.step_heights.get(leg, 0.0)
        bh  = int(bar_maxh * sh / max(max_sh, 1.0))
        col = (0, 80, 255) if leg == result.affected_leg else (0, 180, 80)
        cv2.rectangle(frame,
                      (cx, bar_base - bh),
                      (cx + bar_w, bar_base),
                      col, -1)
        cv2.rectangle(frame,
                      (cx, bar_base - bar_maxh),
                      (cx + bar_w, bar_base),
                      (80, 80, 80), 1)

    return frame


# ── Model loader (cached) ──────────────────────────────────────────────────────

_model_cache: dict = {}

def load_model(model_path: str) -> YOLO:
    if model_path not in _model_cache:
        p = Path(model_path)
        if not p.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        print(f"[model] Loading {p.name}  ({p.stat().st_size/1e6:.1f} MB)")
        _model_cache[model_path] = YOLO(model_path)
    return _model_cache[model_path]


# ── Per-frame inference ────────────────────────────────────────────────────────

def infer_frame(
    frame: np.ndarray,
    model: YOLO,
    box_conf_thresh: float = 0.55,
    box_iou_thresh:  float = 0.45,
    kpt_conf_thresh: float = 0.30,
) -> tuple:
    """
    Single-model inference: detect dogs + 24 keypoints in one pass.
    Returns (annotated_frame, stats_dict, raw_result).
    """
    stats = {"dogs": 0, "keypoints": 0}

    try:
        res = model.predict(
            frame,
            conf    = box_conf_thresh,
            iou     = box_iou_thresh,
            verbose = False,
        )[0].cpu()

        if not len(res.boxes.xyxy):
            return frame, stats, None

        stats["dogs"] = len(res.boxes.xyxy)

        has_kpts = (
            res.keypoints is not None
            and res.keypoints.xy   is not None
            and res.keypoints.conf is not None
            and len(res.keypoints.xy) > 0
        )

        kpts_xy_all   = res.keypoints.xy.numpy()   if has_kpts else None  # (N,24,2)
        kpts_conf_all = res.keypoints.conf.numpy() if has_kpts else None  # (N,24)

        for i, (box, score) in enumerate(zip(
                res.boxes.xyxy.numpy(), res.boxes.conf.numpy())):

            frame = draw_box(frame, box, score=score)

            if has_kpts and i < len(kpts_xy_all):
                frame, n_vis = draw_pose(
                    frame,
                    kpts_xy_all[i],
                    kpts_conf_all[i],
                    kpt_conf_thresh,
                )
                stats["keypoints"] += n_vis

        return frame, stats, res if has_kpts else None

    except Exception as e:
        import traceback
        print(f"[inference] frame error: {e}")
        traceback.print_exc()

    return frame, stats, None


# ── Full video processing ──────────────────────────────────────────────────────

def process_video(
    input_path:      str,
    output_path:     str,
    model_path:      str,
    det_model_path:  str   = None,   # ignored — kept for API compat
    box_conf_thresh: float = 0.55,
    box_iou_thresh:  float = 0.45,
    kpt_conf_thresh: float = 0.30,
    progress_cb             = None,
) -> dict:
    """
    Process a video end-to-end with best.pt (single model, 24 dog keypoints).
    """
    model = load_model(model_path)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {input_path}")

    fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    analyser   = GaitAnalyser(fps=fps, window=WINDOW_FRAMES)
    gait_result = GaitResult.empty()

    frame_idx  = 0
    dog_frames = 0
    total_dogs = 0
    total_kpts = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        frame, stats, raw_res = infer_frame(
            frame, model, box_conf_thresh, box_iou_thresh, kpt_conf_thresh
        )

        # ── Gait analysis: feed best (highest-conf) dog to analyser ───────────
        try:
            if raw_res is not None:
                kc_all = raw_res.keypoints.conf.numpy()
                best   = int(np.argmax(kc_all.mean(axis=1)))
                gait_result = analyser.update(
                    frame_idx,
                    raw_res.keypoints.xy.numpy()[best],
                    kc_all[best],
                )
        except Exception:
            pass

        # ── Draw gait overlay panel ────────────────────────────────────────────
        frame = draw_gait_overlay(frame, gait_result)

        # ── Top HUD ───────────────────────────────────────────────────────────
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, 36), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.45, frame, 0.55, 0)
        cv2.putText(frame,
                    f"Frame {frame_idx + 1}/{total_frames}  |  Dogs: {stats['dogs']}  |  Keypoints: {stats['keypoints']}  |  {gait_result.label}",
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                    (0, 255, 128), 1, cv2.LINE_AA)

        writer.write(frame)

        if stats["dogs"] > 0:
            dog_frames += 1
        total_dogs += stats["dogs"]
        total_kpts += stats["keypoints"]
        frame_idx  += 1

        if progress_cb and total_frames > 0:
            progress_cb(round(frame_idx / total_frames * 100, 1))

    cap.release()
    writer.release()

    out_size = Path(output_path).stat().st_size / 1e6
    return {
        "total_frames":       total_frames,
        "frames_processed":   frame_idx,
        "frames_with_dog":    dog_frames,
        "detection_rate":     round(dog_frames / max(frame_idx, 1) * 100, 1),
        "avg_dogs_per_frame": round(total_dogs / max(frame_idx, 1), 2),
        "avg_kpts_per_frame": round(total_kpts / max(frame_idx, 1), 1),
        "output_size_mb":     round(out_size, 2),
        "gait_assessment":    gait_result.to_dict(),
    }
