"""
main.py — FastAPI backend for RescueAI: AI-Based Stray Dog Injury Detection.

Endpoints:
  POST /auth/login              — authenticate user, returns JWT token
  GET  /auth/me                 — get current user info

  POST /upload                  — upload a video, returns job_id
  GET  /status/{job_id}         — poll processing progress (0-100)
  GET  /result/{job_id}         — download the processed video
  GET  /summary/{job_id}        — get inference stats + gait assessment JSON

  GET  /alerts                  — list injury alerts (filterable by status)
  GET  /alerts/{alert_id}       — get single alert
  PATCH /alerts/{alert_id}      — update alert status (In-Progress / Resolved)

  GET  /cameras                 — list registered cameras
  POST /cameras                 — add a new camera
  PATCH /cameras/{camera_id}    — update camera status
  DELETE /cameras/{camera_id}   — remove a camera

  GET  /stats                   — dashboard statistics
  GET  /hotspots                — per-camera alert counts for heatmap
  GET  /logs                    — system audit logs (admin only)
  GET  /reports/monthly         — monthly summary report
"""

import os
import uuid
import threading
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Header, Depends, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.inference import process_video
from backend.auth import authenticate, verify_token, USERS
from backend.database import db

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR       = Path(__file__).resolve().parent.parent
MODEL_PATH     = str(BASE_DIR / "best.pt")
UPLOAD_DIR     = BASE_DIR / "uploads"
OUTPUT_DIR     = BASE_DIR / "outputs"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# ── In-memory job store ────────────────────────────────────────────────────────
jobs: dict[str, dict] = {}

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(title="RescueAI — Stray Dog Injury Detection API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the frontend
FRONTEND_DIR = BASE_DIR / "frontend"
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


# ── Auth helpers ───────────────────────────────────────────────────────────────

def get_current_user(authorization: str = Header(default=None)) -> dict:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid token")
    token = authorization.split(" ", 1)[1]
    payload = verify_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Token expired or invalid")
    return payload


def require_admin(user: dict = Depends(get_current_user)) -> dict:
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return user


# ── Pydantic models ────────────────────────────────────────────────────────────

class LoginRequest(BaseModel):
    username: str
    password: str


class AlertUpdateRequest(BaseModel):
    status:      str                  # In-Progress | Resolved
    assigned_to: Optional[str] = None
    notes:       Optional[str] = ""


class CameraCreateRequest(BaseModel):
    name:      str
    location:  str
    latitude:  float
    longitude: float
    rtsp_url:  str


class CameraStatusRequest(BaseModel):
    status: str   # Active | Inactive | Maintenance


# ── Background worker ──────────────────────────────────────────────────────────

def _run_inference(job_id: str, input_path: str, output_path: str,
                   camera_id: str = "UPLOAD", camera_name: str = "Manual Upload",
                   location: str = "Unknown", lat: float = 0.0, lon: float = 0.0):
    jobs[job_id]["status"] = "processing"

    def _progress(pct: float):
        jobs[job_id]["progress"] = pct

    try:
        summary = process_video(
            input_path      = input_path,
            output_path     = output_path,
            model_path      = MODEL_PATH,
            det_model_path  = None,
            box_conf_thresh = 0.55,
            box_iou_thresh  = 0.45,
            kpt_conf_thresh = 0.30,
            progress_cb     = _progress,
        )
        jobs[job_id]["status"]   = "done"
        jobs[job_id]["progress"] = 100
        jobs[job_id]["summary"]  = summary

        # ── Auto-create alert if limping detected ──────────────────────────────
        gait = summary.get("gait_assessment", {})
        if gait.get("is_limping"):
            alert = db.add_alert(
                camera_id   = camera_id,
                camera_name = camera_name,
                location    = location,
                latitude    = lat,
                longitude   = lon,
                severity    = gait.get("severity", "Mild"),
                affected_leg= gait.get("affected_leg"),
                confidence  = round(summary.get("detection_rate", 0) / 100, 2),
                video_clip  = output_path,
                job_id      = job_id,
                gait_data   = gait,
            )
            jobs[job_id]["alert_id"] = alert.alert_id
            db.log("system", "Injury Alert Created",
                   f"Alert {alert.alert_id} — {gait.get('label')} at {camera_name}")

    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"]  = str(e)
        db.log("system", "Inference Error", str(e), level="ERROR")
        print(f"[job {job_id}] error: {e}")


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/static/index.html")


# ── Auth ───────────────────────────────────────────────────────────────────────

@app.post("/auth/login")
def login(req: LoginRequest):
    user = authenticate(req.username, req.password)
    if not user:
        db.log(req.username, "Login Failed", "Invalid credentials", level="WARNING")
        raise HTTPException(status_code=401, detail="Invalid username or password")
    from backend.auth import create_token
    token = create_token(user["username"], user["role"])
    db.log(user["username"], "Login", f"{user['name']} logged in")
    return JSONResponse({
        "token":    token,
        "username": user["username"],
        "role":     user["role"],
        "name":     user["name"],
        "ngo":      user["ngo"],
    })


@app.get("/auth/me")
def me(user: dict = Depends(get_current_user)):
    username = user["sub"]
    u = USERS.get(username, {})
    return JSONResponse({
        "username": username,
        "role":     user["role"],
        "name":     u.get("name", username),
        "ngo":      u.get("ngo", ""),
    })


# ── Video Upload & Inference ───────────────────────────────────────────────────

@app.post("/upload")
async def upload_video(
    file:        UploadFile = File(...),
    camera_id:   str = Query(default="UPLOAD"),
    camera_name: str = Query(default="Manual Upload"),
    location:    str = Query(default="Unknown"),
    lat:         float = Query(default=0.0),
    lon:         float = Query(default=0.0),
    authorization: str = Header(default=None),
):
    allowed = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    suffix  = Path(file.filename).suffix.lower()
    if suffix not in allowed:
        raise HTTPException(status_code=400,
                            detail=f"Unsupported file type '{suffix}'. Allowed: {allowed}")

    job_id      = str(uuid.uuid4())
    input_path  = str(UPLOAD_DIR / f"{job_id}_input{suffix}")
    output_path = str(OUTPUT_DIR / f"{job_id}_output.mp4")

    content = await file.read()
    with open(input_path, "wb") as f:
        f.write(content)

    jobs[job_id] = {
        "status":    "queued",
        "progress":  0,
        "summary":   None,
        "error":     None,
        "alert_id":  None,
        "input":     input_path,
        "output":    output_path,
        "uploaded_by": "anonymous",
    }

    t = threading.Thread(
        target=_run_inference,
        args=(job_id, input_path, output_path, camera_id, camera_name, location, lat, lon),
        daemon=True,
    )
    t.start()

    db.log("anonymous", "Video Uploaded", f"Job {job_id} — {file.filename} ({len(content)/1e6:.1f} MB)")
    return JSONResponse({"job_id": job_id})


@app.get("/status/{job_id}")
def get_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = jobs[job_id]
    return JSONResponse({
        "status":   job["status"],
        "progress": job["progress"],
        "error":    job["error"],
        "alert_id": job.get("alert_id"),
    })


@app.get("/result/{job_id}")
def get_result(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = jobs[job_id]
    if job["status"] != "done":
        raise HTTPException(status_code=400, detail=f"Job not ready: {job['status']}")
    output_path = job["output"]
    if not Path(output_path).exists():
        raise HTTPException(status_code=404, detail="Output file missing")
    return FileResponse(output_path, media_type="video/mp4",
                        filename=f"rescueai_{job_id[:8]}.mp4")


@app.get("/summary/{job_id}")
def get_summary(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = jobs[job_id]
    if job["status"] != "done":
        raise HTTPException(status_code=400, detail=f"Job not ready: {job['status']}")
    return JSONResponse(job["summary"])


# ── Alerts ─────────────────────────────────────────────────────────────────────

@app.get("/alerts")
def list_alerts(
    status: Optional[str] = Query(default=None),
    limit:  int           = Query(default=50),
    user: dict = Depends(get_current_user),
):
    alerts = db.get_alerts(status=status, limit=limit)
    return JSONResponse([a.to_dict() for a in alerts])


@app.get("/alerts/{alert_id}")
def get_alert(alert_id: str, user: dict = Depends(get_current_user)):
    alert = db.get_alert(alert_id)
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    return JSONResponse(alert.to_dict())


@app.patch("/alerts/{alert_id}")
def update_alert(alert_id: str, req: AlertUpdateRequest,
                 user: dict = Depends(get_current_user)):
    valid_statuses = {"New", "In-Progress", "Resolved"}
    if req.status not in valid_statuses:
        raise HTTPException(status_code=400, detail=f"Invalid status. Must be one of {valid_statuses}")
    alert = db.update_alert_status(
        alert_id    = alert_id,
        status      = req.status,
        assigned_to = req.assigned_to or user["sub"],
        notes       = req.notes or "",
    )
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    db.log(user["sub"], "Alert Updated",
           f"Alert {alert_id} → {req.status} by {user['sub']}")
    return JSONResponse(alert.to_dict())


# ── Cameras ────────────────────────────────────────────────────────────────────

@app.get("/cameras")
def list_cameras(user: dict = Depends(get_current_user)):
    return JSONResponse([c.to_dict() for c in db.cameras])


@app.post("/cameras")
def add_camera(req: CameraCreateRequest, user: dict = require_admin):
    cam = db.add_camera(
        name      = req.name,
        location  = req.location,
        latitude  = req.latitude,
        longitude = req.longitude,
        rtsp_url  = req.rtsp_url,
        added_by  = user["sub"] if isinstance(user, dict) else "admin",
    )
    db.log("admin", "Camera Added", f"Camera {cam.camera_id} — {cam.name}")
    return JSONResponse(cam.to_dict())


@app.patch("/cameras/{camera_id}")
def update_camera(camera_id: str, req: CameraStatusRequest,
                  user: dict = Depends(require_admin)):
    cam = db.update_camera_status(camera_id, req.status)
    if not cam:
        raise HTTPException(status_code=404, detail="Camera not found")
    db.log(user["sub"], "Camera Updated", f"Camera {camera_id} → {req.status}")
    return JSONResponse(cam.to_dict())


@app.delete("/cameras/{camera_id}")
def delete_camera(camera_id: str, user: dict = Depends(require_admin)):
    if not db.delete_camera(camera_id):
        raise HTTPException(status_code=404, detail="Camera not found")
    db.log(user["sub"], "Camera Deleted", f"Camera {camera_id} removed")
    return JSONResponse({"message": "Camera deleted"})


# ── Dashboard ──────────────────────────────────────────────────────────────────

@app.get("/stats")
def get_stats(user: dict = Depends(get_current_user)):
    return JSONResponse(db.get_stats())


@app.get("/hotspots")
def get_hotspots(user: dict = Depends(get_current_user)):
    return JSONResponse(db.get_hotspot_data())


@app.get("/logs")
def get_logs(limit: int = Query(default=100), user: dict = Depends(require_admin)):
    return JSONResponse([l.to_dict() for l in db.get_logs(limit=limit)])


@app.get("/reports/monthly")
def monthly_report(user: dict = Depends(get_current_user)):
    """Aggregate alerts by day for the last 30 days."""
    now = time.time()
    thirty_days_ago = now - 30 * 86400
    recent = [a for a in db.alerts if a.timestamp >= thirty_days_ago]

    # Group by date
    by_date: dict[str, dict] = {}
    for a in recent:
        date_str = time.strftime("%Y-%m-%d", time.localtime(a.timestamp))
        if date_str not in by_date:
            by_date[date_str] = {"date": date_str, "total": 0, "severe": 0, "resolved": 0}
        by_date[date_str]["total"] += 1
        if a.severity == "Severe":
            by_date[date_str]["severe"] += 1
        if a.status == "Resolved":
            by_date[date_str]["resolved"] += 1

    return JSONResponse({
        "period":  "Last 30 days",
        "summary": db.get_stats(),
        "by_date": sorted(by_date.values(), key=lambda x: x["date"]),
    })
