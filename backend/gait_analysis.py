"""
gait_analysis.py — Biomechanical gait analysis and limping detection.

Analyses per-frame keypoint trajectories to detect:
  1. Step height asymmetry  — a limping leg lifts less than its contralateral
  2. Stance time asymmetry  — a limping leg spends more time on the ground
  3. Symmetry Index (SI)    — standard clinical metric, >10% = asymmetric gait
  4. Joint angle reduction  — reduced ROM in the injured limb
  5. Head-bob               — compensatory head movement (front-leg lameness)
  6. Hip-hike               — compensatory hip rise (rear-leg lameness)

Keypoint indices (from StanfordExtra / keypoint_definitions.csv):
  0  L_Front_Paw    1  L_Front_Knee   2  L_Front_Elbow
  3  L_Rear_Paw     4  L_Rear_Knee    5  L_Rear_Elbow
  6  R_Front_Paw    7  R_Front_Knee   8  R_Front_Elbow
  9  R_Rear_Paw    10  R_Rear_Knee   11  R_Rear_Elbow
 12  Tail_Base     13  Tail_End
 14  L_Ear_Base    15  R_Ear_Base
 16  Nose          17  Chin
 18  L_Ear_Tip     19  R_Ear_Tip
 20  L_Eye         21  R_Eye
 22  Withers       23  Throat
"""

from __future__ import annotations
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

# ── Keypoint index constants ───────────────────────────────────────────────────
LF_PAW, LF_KNEE, LF_ELBOW = 0, 1, 2
LR_PAW, LR_KNEE, LR_ELBOW = 3, 4, 5
RF_PAW, RF_KNEE, RF_ELBOW = 6, 7, 8
RR_PAW, RR_KNEE, RR_ELBOW = 9, 10, 11
TAIL_BASE = 12
NOSE      = 16
WITHERS   = 22
THROAT    = 23

PAW_IDS   = {"LF": LF_PAW,   "LR": LR_PAW,   "RF": RF_PAW,   "RR": RR_PAW}
KNEE_IDS  = {"LF": LF_KNEE,  "LR": LR_KNEE,  "RF": RF_KNEE,  "RR": RR_KNEE}
ELBOW_IDS = {"LF": LF_ELBOW, "LR": LR_ELBOW, "RF": RF_ELBOW, "RR": RR_ELBOW}

LEG_PAIRS = [("LF", "RF"), ("LR", "RR")]   # contralateral pairs

# ── Thresholds (tuned on the gait assessment video) ───────────────────────────
SI_LIMP_THRESHOLD      = 15.0   # Symmetry Index % above which = asymmetric
SI_SEVERE_THRESHOLD    = 30.0   # SI % above which = severe
STANCE_DIFF_THRESHOLD  = 0.08   # stance ratio difference > 8% = asymmetric
ANGLE_DIFF_THRESHOLD   = 20.0   # mean joint angle difference > 20° = reduced ROM
HEAD_BOB_THRESHOLD     = 8.0    # nose Y std > 8px relative to withers = head bob
HIP_HIKE_THRESHOLD     = 8.0    # tail-base Y std > 8px relative to withers = hip hike
MIN_CONF               = 0.30   # minimum keypoint confidence to use
WINDOW_FRAMES          = 30     # rolling window size for metrics


# ── Helpers ────────────────────────────────────────────────────────────────────

def _angle_3pts(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle at point b formed by a-b-c, in degrees."""
    ba = a - b
    bc = c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8
    cos_a = np.dot(ba, bc) / denom
    return float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))


def _symmetry_index(val_l: float, val_r: float) -> float:
    """
    Standard Symmetry Index:  |L - R| / ((L + R) / 2) * 100
    0 = perfect symmetry, higher = more asymmetric.
    """
    denom = (val_l + val_r) / 2.0
    if denom < 1e-6:
        return 0.0
    return abs(val_l - val_r) / denom * 100.0


# ── Per-frame data ─────────────────────────────────────────────────────────────

@dataclass
class FrameKpts:
    """Keypoints for one dog instance in one frame."""
    frame_idx: int
    xy:   np.ndarray   # (24, 2)
    conf: np.ndarray   # (24,)

    def get(self, kid: int) -> Optional[np.ndarray]:
        """Return (x, y) if confidence is sufficient, else None."""
        if self.conf[kid] >= MIN_CONF:
            return self.xy[kid].copy()
        return None

    def paw_y(self, leg: str) -> Optional[float]:
        pt = self.get(PAW_IDS[leg])
        return float(pt[1]) if pt is not None else None

    def knee_angle(self, leg: str) -> Optional[float]:
        a = self.get(PAW_IDS[leg])
        b = self.get(KNEE_IDS[leg])
        c = self.get(ELBOW_IDS[leg])
        if a is None or b is None or c is None:
            return None
        return _angle_3pts(a, b, c)


# ── Rolling window analyser ────────────────────────────────────────────────────

@dataclass
class LegMetrics:
    step_heights:  list = field(default_factory=list)   # paw Y range per window
    stance_ratios: list = field(default_factory=list)   # fraction near ground
    knee_angles:   list = field(default_factory=list)   # joint angles


class GaitAnalyser:
    """
    Accumulates per-frame keypoints and computes rolling gait metrics.
    Call update() each frame, read result() for the current assessment.
    """

    def __init__(self, fps: float = 30.0, window: int = WINDOW_FRAMES):
        self.fps    = fps
        self.window = window
        self._buf: deque[FrameKpts] = deque(maxlen=window)

        # Running history for trend detection
        self._si_history:    deque[dict] = deque(maxlen=90)   # ~3 s
        self._angle_history: deque[dict] = deque(maxlen=90)

    # ── Public API ─────────────────────────────────────────────────────────────

    def update(self, frame_idx: int,
               kpts_xy: np.ndarray,
               kpts_conf: np.ndarray) -> "GaitResult":
        """
        Feed one frame's keypoints (for the primary/best dog).
        Returns a GaitResult with the current assessment.
        """
        fk = FrameKpts(frame_idx, kpts_xy, kpts_conf)
        self._buf.append(fk)
        return self._analyse()

    def reset(self):
        self._buf.clear()
        self._si_history.clear()
        self._angle_history.clear()

    # ── Internal analysis ──────────────────────────────────────────────────────

    def _analyse(self) -> "GaitResult":
        buf = list(self._buf)
        if len(buf) < 5:
            return GaitResult.empty()

        # ── 1. Step height per leg ─────────────────────────────────────────────
        step_heights: dict[str, float] = {}
        for leg in PAW_IDS:
            ys = [f.paw_y(leg) for f in buf]
            ys = [y for y in ys if y is not None]
            step_heights[leg] = float(np.ptp(ys)) if len(ys) > 3 else 0.0

        # ── 2. Stance ratio per leg ────────────────────────────────────────────
        stance_ratios: dict[str, float] = {}
        for leg in PAW_IDS:
            ys = [f.paw_y(leg) for f in buf]
            ys = [y for y in ys if y is not None]
            if len(ys) < 3:
                stance_ratios[leg] = 0.5
                continue
            ys_arr = np.array(ys)
            ground_thresh = ys_arr.mean() - 0.3 * ys_arr.std()
            stance_ratios[leg] = float((ys_arr >= ground_thresh).mean())

        # ── 3. Mean knee angle per leg ─────────────────────────────────────────
        knee_angles: dict[str, float] = {}
        for leg in PAW_IDS:
            angles = [f.knee_angle(leg) for f in buf]
            angles = [a for a in angles if a is not None]
            knee_angles[leg] = float(np.mean(angles)) if angles else 180.0

        # ── 4. Head bob (nose Y relative to withers) ───────────────────────────
        nose_rel_ys = []
        for f in buf:
            n = f.get(NOSE)
            w = f.get(WITHERS)
            if n is not None and w is not None:
                nose_rel_ys.append(float(n[1] - w[1]))
        head_bob = float(np.std(nose_rel_ys)) if len(nose_rel_ys) > 3 else 0.0

        # ── 5. Hip hike (tail-base Y relative to withers) ─────────────────────
        hip_rel_ys = []
        for f in buf:
            t = f.get(TAIL_BASE)
            w = f.get(WITHERS)
            if t is not None and w is not None:
                hip_rel_ys.append(float(t[1] - w[1]))
        hip_hike = float(np.std(hip_rel_ys)) if len(hip_rel_ys) > 3 else 0.0

        # ── 6. Symmetry indices ────────────────────────────────────────────────
        si_step:   dict[str, float] = {}
        si_stance: dict[str, float] = {}
        si_angle:  dict[str, float] = {}
        for l_leg, r_leg in LEG_PAIRS:
            pair = f"{l_leg}/{r_leg}"
            si_step[pair]   = _symmetry_index(step_heights[l_leg],  step_heights[r_leg])
            si_stance[pair] = _symmetry_index(stance_ratios[l_leg], stance_ratios[r_leg])
            si_angle[pair]  = _symmetry_index(knee_angles[l_leg],   knee_angles[r_leg])

        # ── 7. Identify the most likely affected leg ───────────────────────────
        affected_leg, severity, reasons = self._identify_limping(
            step_heights, stance_ratios, knee_angles,
            si_step, si_stance, si_angle,
            head_bob, hip_hike,
        )

        return GaitResult(
            affected_leg  = affected_leg,
            severity      = severity,
            reasons       = reasons,
            step_heights  = step_heights,
            stance_ratios = stance_ratios,
            knee_angles   = knee_angles,
            si_step       = si_step,
            si_stance     = si_stance,
            si_angle      = si_angle,
            head_bob      = head_bob,
            hip_hike      = hip_hike,
            n_frames      = len(buf),
        )

    def _identify_limping(
        self,
        step_heights:  dict,
        stance_ratios: dict,
        knee_angles:   dict,
        si_step:       dict,
        si_stance:     dict,
        si_angle:      dict,
        head_bob:      float,
        hip_hike:      float,
    ) -> tuple[Optional[str], str, list[str]]:
        """
        Score each leg for lameness evidence.
        Returns (affected_leg | None, severity_label, [reason strings]).
        """
        scores: dict[str, float] = {leg: 0.0 for leg in PAW_IDS}
        reasons: list[str] = []

        # ── Step height: lower lift = more likely injured ──────────────────────
        for l_leg, r_leg in LEG_PAIRS:
            pair = f"{l_leg}/{r_leg}"
            si   = si_step[pair]
            if si > SI_LIMP_THRESHOLD:
                # The leg with LESS lift is the suspect
                if step_heights[l_leg] < step_heights[r_leg]:
                    scores[l_leg] += si / 10.0
                    reasons.append(
                        f"Low step lift: {l_leg} ({step_heights[l_leg]:.1f}px) "
                        f"vs {r_leg} ({step_heights[r_leg]:.1f}px) — SI={si:.1f}%"
                    )
                else:
                    scores[r_leg] += si / 10.0
                    reasons.append(
                        f"Low step lift: {r_leg} ({step_heights[r_leg]:.1f}px) "
                        f"vs {l_leg} ({step_heights[l_leg]:.1f}px) — SI={si:.1f}%"
                    )

        # ── Stance ratio: longer ground contact = more likely injured ──────────
        for l_leg, r_leg in LEG_PAIRS:
            pair = f"{l_leg}/{r_leg}"
            diff = abs(stance_ratios[l_leg] - stance_ratios[r_leg])
            if diff > STANCE_DIFF_THRESHOLD:
                # The leg with MORE stance time is the suspect
                if stance_ratios[l_leg] > stance_ratios[r_leg]:
                    scores[l_leg] += diff * 10.0
                    reasons.append(
                        f"Extended stance: {l_leg} ({stance_ratios[l_leg]:.2f}) "
                        f"vs {r_leg} ({stance_ratios[r_leg]:.2f})"
                    )
                else:
                    scores[r_leg] += diff * 10.0
                    reasons.append(
                        f"Extended stance: {r_leg} ({stance_ratios[r_leg]:.2f}) "
                        f"vs {l_leg} ({stance_ratios[l_leg]:.2f})"
                    )

        # ── Joint angle: reduced ROM = more likely injured ─────────────────────
        for l_leg, r_leg in LEG_PAIRS:
            pair = f"{l_leg}/{r_leg}"
            diff = abs(knee_angles[l_leg] - knee_angles[r_leg])
            if diff > ANGLE_DIFF_THRESHOLD:
                if knee_angles[l_leg] < knee_angles[r_leg]:
                    scores[l_leg] += diff / 20.0
                    reasons.append(
                        f"Reduced ROM: {l_leg} knee {knee_angles[l_leg]:.1f}° "
                        f"vs {r_leg} {knee_angles[r_leg]:.1f}°"
                    )
                else:
                    scores[r_leg] += diff / 20.0
                    reasons.append(
                        f"Reduced ROM: {r_leg} knee {knee_angles[r_leg]:.1f}° "
                        f"vs {l_leg} {knee_angles[l_leg]:.1f}°"
                    )

        # ── Compensatory signs ─────────────────────────────────────────────────
        if head_bob > HEAD_BOB_THRESHOLD:
            reasons.append(f"Head bob detected (nose std={head_bob:.1f}px) — front leg suspect")
            scores["LF"] += head_bob / 20.0
            scores["RF"] += head_bob / 20.0

        if hip_hike > HIP_HIKE_THRESHOLD:
            reasons.append(f"Hip hike detected (hip std={hip_hike:.1f}px) — rear leg suspect")
            scores["LR"] += hip_hike / 20.0
            scores["RR"] += hip_hike / 20.0

        # ── Determine affected leg and severity ────────────────────────────────
        max_score = max(scores.values())

        if max_score < 0.5:
            return None, "Normal", []

        affected_leg = max(scores, key=scores.__getitem__)

        # Severity based on max SI across all pairs
        max_si = max(
            list(si_step.values()) +
            list(si_stance.values()) +
            list(si_angle.values())
        )
        if max_si >= SI_SEVERE_THRESHOLD or max_score >= 3.0:
            severity = "Severe"
        elif max_si >= SI_LIMP_THRESHOLD or max_score >= 1.5:
            severity = "Moderate"
        else:
            severity = "Mild"

        return affected_leg, severity, reasons


# ── Result dataclass ───────────────────────────────────────────────────────────

@dataclass
class GaitResult:
    affected_leg:  Optional[str]        # "LF" | "LR" | "RF" | "RR" | None
    severity:      str                  # "Normal" | "Mild" | "Moderate" | "Severe"
    reasons:       list[str]
    step_heights:  dict[str, float]
    stance_ratios: dict[str, float]
    knee_angles:   dict[str, float]
    si_step:       dict[str, float]
    si_stance:     dict[str, float]
    si_angle:      dict[str, float]
    head_bob:      float
    hip_hike:      float
    n_frames:      int

    @staticmethod
    def empty() -> "GaitResult":
        legs = {"LF": 0.0, "LR": 0.0, "RF": 0.0, "RR": 0.0}
        pairs = {"LF/RF": 0.0, "LR/RR": 0.0}
        return GaitResult(
            affected_leg  = None,
            severity      = "Analysing...",
            reasons       = [],
            step_heights  = legs.copy(),
            stance_ratios = {k: 0.5 for k in legs},
            knee_angles   = {k: 180.0 for k in legs},
            si_step       = pairs.copy(),
            si_stance     = pairs.copy(),
            si_angle      = pairs.copy(),
            head_bob      = 0.0,
            hip_hike      = 0.0,
            n_frames      = 0,
        )

    @property
    def is_limping(self) -> bool:
        return self.severity not in ("Normal", "Analysing...")

    @property
    def label(self) -> str:
        if self.severity == "Analysing...":
            return "Analysing..."
        if not self.is_limping:
            return "Normal Gait"
        leg_full = {
            "LF": "Left Front", "LR": "Left Rear",
            "RF": "Right Front", "RR": "Right Rear",
        }
        leg_name = leg_full.get(self.affected_leg, self.affected_leg)
        return f"{self.severity} Limp — {leg_name}"

    def to_dict(self) -> dict:
        return {
            "affected_leg":  self.affected_leg,
            "severity":      self.severity,
            "label":         self.label,
            "is_limping":    self.is_limping,
            "reasons":       self.reasons,
            "step_heights":  self.step_heights,
            "stance_ratios": {k: round(v, 3) for k, v in self.stance_ratios.items()},
            "knee_angles":   {k: round(v, 1) for k, v in self.knee_angles.items()},
            "si_step":       {k: round(v, 1) for k, v in self.si_step.items()},
            "si_stance":     {k: round(v, 1) for k, v in self.si_stance.items()},
            "si_angle":      {k: round(v, 1) for k, v in self.si_angle.items()},
            "head_bob":      round(self.head_bob, 1),
            "hip_hike":      round(self.hip_hike, 1),
            "n_frames":      self.n_frames,
        }
