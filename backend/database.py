"""
database.py — In-memory data store for RescueAI.

Stores:
  - injury_alerts: detected injury events
  - cameras: registered CCTV cameras
  - system_logs: audit trail
  - rescue_stats: aggregated statistics

Replace with MySQL/PostgreSQL in production.
"""

from __future__ import annotations
import uuid
import time
from typing import Optional
from dataclasses import dataclass, field, asdict


# ── Alert ──────────────────────────────────────────────────────────────────────

@dataclass
class InjuryAlert:
    alert_id:       str
    timestamp:      float
    camera_id:      str
    camera_name:    str
    location:       str
    latitude:       float
    longitude:      float
    severity:       str          # Normal | Mild | Moderate | Severe
    affected_leg:   Optional[str]
    confidence:     float
    video_clip:     Optional[str]   # path to short clip
    job_id:         Optional[str]
    status:         str = "New"     # New | In-Progress | Resolved
    assigned_to:    Optional[str] = None
    resolved_at:    Optional[float] = None
    notes:          str = ""
    gait_data:      dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["timestamp_str"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.timestamp))
        return d


# ── Camera ─────────────────────────────────────────────────────────────────────

@dataclass
class Camera:
    camera_id:   str
    name:        str
    location:    str
    latitude:    float
    longitude:   float
    rtsp_url:    str
    status:      str = "Active"   # Active | Inactive | Maintenance
    added_by:    str = "admin"
    added_at:    float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)


# ── System Log ─────────────────────────────────────────────────────────────────

@dataclass
class SystemLog:
    log_id:    str
    timestamp: float
    user:      str
    action:    str
    details:   str
    level:     str = "INFO"   # INFO | WARNING | ERROR

    def to_dict(self) -> dict:
        d = asdict(self)
        d["timestamp_str"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.timestamp))
        return d


# ── In-memory DB ───────────────────────────────────────────────────────────────

class Database:
    def __init__(self):
        self.alerts:  list[InjuryAlert] = []
        self.cameras: list[Camera]      = []
        self.logs:    list[SystemLog]   = []
        self._seed()

    def _seed(self):
        """Seed with demo cameras and sample alerts."""
        demo_cameras = [
            Camera("CAM-001", "Main Street Junction",    "Main St & 2nd Ave, Lahore",  31.5204, 74.3587, "rtsp://cam001.safecity.pk/live"),
            Camera("CAM-002", "Gulberg Market",          "Gulberg III, Lahore",         31.5100, 74.3450, "rtsp://cam002.safecity.pk/live"),
            Camera("CAM-003", "DHA Phase 5 Roundabout", "DHA Phase 5, Lahore",         31.4700, 74.4100, "rtsp://cam003.safecity.pk/live"),
            Camera("CAM-004", "Johar Town Park",         "Johar Town, Lahore",          31.4690, 74.2810, "rtsp://cam004.safecity.pk/live", status="Inactive"),
            Camera("CAM-005", "Model Town Chowk",        "Model Town, Lahore",          31.4800, 74.3200, "rtsp://cam005.safecity.pk/live"),
        ]
        self.cameras.extend(demo_cameras)

        # Sample historical alerts
        now = time.time()
        sample_alerts = [
            InjuryAlert("ALT-001", now - 7200,  "CAM-001", "Main Street Junction",    "Main St & 2nd Ave, Lahore",  31.5204, 74.3587, "Severe",   "LF",  0.91, None, None, "Resolved",     "ngo1", now - 6800, "Dog rescued and taken to vet.", {"si_step": {"LF/RF": 42.3}}),
            InjuryAlert("ALT-002", now - 3600,  "CAM-002", "Gulberg Market",          "Gulberg III, Lahore",         31.5100, 74.3450, "Moderate", "RR",  0.78, None, None, "In-Progress",  "ngo2", None,       "Team dispatched.", {"si_step": {"LR/RR": 28.1}}),
            InjuryAlert("ALT-003", now - 1800,  "CAM-003", "DHA Phase 5 Roundabout", "DHA Phase 5, Lahore",         31.4700, 74.4100, "Mild",     "RF",  0.65, None, None, "New",          None,   None,       "", {"si_step": {"LF/RF": 17.5}}),
            InjuryAlert("ALT-004", now - 900,   "CAM-005", "Model Town Chowk",        "Model Town, Lahore",          31.4800, 74.3200, "Severe",   "LR",  0.88, None, None, "New",          None,   None,       "", {"si_step": {"LR/RR": 38.9}}),
            InjuryAlert("ALT-005", now - 300,   "CAM-001", "Main Street Junction",    "Main St & 2nd Ave, Lahore",  31.5204, 74.3587, "Moderate", "RF",  0.72, None, None, "New",          None,   None,       "", {"si_step": {"LF/RF": 25.6}}),
        ]
        self.alerts.extend(sample_alerts)

        self.log("system", "System started", "RescueAI backend initialized with seed data.")

    # ── Alert methods ──────────────────────────────────────────────────────────

    def add_alert(self, **kwargs) -> InjuryAlert:
        alert = InjuryAlert(
            alert_id=f"ALT-{str(uuid.uuid4())[:8].upper()}",
            timestamp=time.time(),
            **kwargs,
        )
        self.alerts.append(alert)
        return alert

    def get_alert(self, alert_id: str) -> Optional[InjuryAlert]:
        return next((a for a in self.alerts if a.alert_id == alert_id), None)

    def update_alert_status(self, alert_id: str, status: str,
                             assigned_to: str = None, notes: str = "") -> Optional[InjuryAlert]:
        alert = self.get_alert(alert_id)
        if not alert:
            return None
        alert.status = status
        if assigned_to:
            alert.assigned_to = assigned_to
        if notes:
            alert.notes = notes
        if status == "Resolved":
            alert.resolved_at = time.time()
        return alert

    def get_alerts(self, status: str = None, limit: int = 100) -> list[InjuryAlert]:
        alerts = self.alerts
        if status:
            alerts = [a for a in alerts if a.status == status]
        return sorted(alerts, key=lambda a: a.timestamp, reverse=True)[:limit]

    # ── Camera methods ─────────────────────────────────────────────────────────

    def add_camera(self, **kwargs) -> Camera:
        cam = Camera(
            camera_id=f"CAM-{str(uuid.uuid4())[:6].upper()}",
            added_at=time.time(),
            **kwargs,
        )
        self.cameras.append(cam)
        return cam

    def get_camera(self, camera_id: str) -> Optional[Camera]:
        return next((c for c in self.cameras if c.camera_id == camera_id), None)

    def update_camera_status(self, camera_id: str, status: str) -> Optional[Camera]:
        cam = self.get_camera(camera_id)
        if cam:
            cam.status = status
        return cam

    def delete_camera(self, camera_id: str) -> bool:
        before = len(self.cameras)
        self.cameras = [c for c in self.cameras if c.camera_id != camera_id]
        return len(self.cameras) < before

    # ── Log methods ────────────────────────────────────────────────────────────

    def log(self, user: str, action: str, details: str, level: str = "INFO"):
        entry = SystemLog(
            log_id=str(uuid.uuid4())[:8],
            timestamp=time.time(),
            user=user,
            action=action,
            details=details,
            level=level,
        )
        self.logs.append(entry)
        # Keep last 500 logs
        if len(self.logs) > 500:
            self.logs = self.logs[-500:]

    def get_logs(self, limit: int = 100) -> list[SystemLog]:
        return sorted(self.logs, key=lambda l: l.timestamp, reverse=True)[:limit]

    # ── Statistics ─────────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        total    = len(self.alerts)
        new      = sum(1 for a in self.alerts if a.status == "New")
        progress = sum(1 for a in self.alerts if a.status == "In-Progress")
        resolved = sum(1 for a in self.alerts if a.status == "Resolved")
        severe   = sum(1 for a in self.alerts if a.severity == "Severe")
        active_cams = sum(1 for c in self.cameras if c.status == "Active")

        # Hotspot: camera with most alerts
        cam_counts: dict[str, int] = {}
        for a in self.alerts:
            cam_counts[a.camera_name] = cam_counts.get(a.camera_name, 0) + 1
        hotspot = max(cam_counts, key=cam_counts.get) if cam_counts else "N/A"

        return {
            "total_alerts":    total,
            "new_alerts":      new,
            "in_progress":     progress,
            "resolved":        resolved,
            "severe_alerts":   severe,
            "active_cameras":  active_cams,
            "total_cameras":   len(self.cameras),
            "hotspot_camera":  hotspot,
            "resolution_rate": round(resolved / max(total, 1) * 100, 1),
        }

    def get_hotspot_data(self) -> list[dict]:
        """Return per-camera alert counts for heatmap."""
        cam_data: dict[str, dict] = {}
        for a in self.alerts:
            if a.camera_id not in cam_data:
                cam = self.get_camera(a.camera_id)
                cam_data[a.camera_id] = {
                    "camera_id":   a.camera_id,
                    "camera_name": a.camera_name,
                    "location":    a.location,
                    "latitude":    a.latitude,
                    "longitude":   a.longitude,
                    "total":       0,
                    "severe":      0,
                    "resolved":    0,
                }
            cam_data[a.camera_id]["total"] += 1
            if a.severity == "Severe":
                cam_data[a.camera_id]["severe"] += 1
            if a.status == "Resolved":
                cam_data[a.camera_id]["resolved"] += 1
        return list(cam_data.values())


# ── Singleton ──────────────────────────────────────────────────────────────────
db = Database()
