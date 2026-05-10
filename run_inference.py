"""
run_inference.py — One-shot inference using best.pt (24 dog keypoints).
"""
import sys
sys.path.insert(0, ".")

from backend.inference import process_video

INPUT_PATH      = "Dogs In Motion Gait Assessment - Micki.mp4"
OUTPUT_PATH     = "output_dog_pose.mp4"
POSE_MODEL_PATH = "best.pt"

def progress(pct):
    bar    = int(pct / 2)
    filled = "#" * bar
    empty  = " " * (50 - bar)
    print(f"\r  [{filled}{empty}] {pct:.1f}%", end="", flush=True)

print("=" * 58)
print("  Dog Pose Inference  [best.pt — 24 dog keypoints]")
print("=" * 58)
print(f"  Input  : {INPUT_PATH}")
print(f"  Output : {OUTPUT_PATH}")
print(f"  Model  : {POSE_MODEL_PATH}")
print("=" * 58)
print()

summary = process_video(
    input_path      = INPUT_PATH,
    output_path     = OUTPUT_PATH,
    model_path      = POSE_MODEL_PATH,
    box_conf_thresh = 0.55,
    box_iou_thresh  = 0.45,
    kpt_conf_thresh = 0.30,
    progress_cb     = progress,
)

print()
print()
print("=" * 58)
print("  INFERENCE COMPLETE")
print("=" * 58)
print(f"  Total frames        : {summary['total_frames']}")
print(f"  Frames processed    : {summary['frames_processed']}")
print(f"  Frames with dog     : {summary['frames_with_dog']}")
print(f"  Detection rate      : {summary['detection_rate']}%")
print(f"  Avg dogs/frame      : {summary['avg_dogs_per_frame']}")
print(f"  Avg keypoints/frame : {summary['avg_kpts_per_frame']}")
print(f"  Output size         : {summary['output_size_mb']} MB")
print(f"  Saved to            : {OUTPUT_PATH}")
print("=" * 58)
print()
print("=== GAIT ASSESSMENT ===")
g = summary.get("gait_assessment", {})
print(f"  Result    : {g.get('label', 'N/A')}")
print(f"  Severity  : {g.get('severity', 'N/A')}")
print(f"  Affected  : {g.get('affected_leg', 'None')}")
print(f"  Head bob  : {g.get('head_bob', 0):.1f}px")
print(f"  Hip hike  : {g.get('hip_hike', 0):.1f}px")
print(f"  SI step   : {g.get('si_step', {})}")
print(f"  SI stance : {g.get('si_stance', {})}")
print("  Reasons:")
for r in g.get("reasons", []):
    print(f"    - {r}")
