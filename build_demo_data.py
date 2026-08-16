"""
Packages the output of a completed run_tracking.py run into demo_data/<slug>/,
in a browser-friendly format, for the hosted (no-GPU, no-API-key) demo app.

Run this right after each real pipeline run completes, before starting the
next one - results/, data/output/tracked_dwell.mp4 and the confusion segment
clips all get overwritten by the following run.

Usage:
    python build_demo_data.py <slug> <label> <original_video_path>
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from config import Config

DEMO_DATA_DIR = Path("demo_data")
MAX_WEB_WIDTH = 1280


def reencode_for_web(src: Path, dst: Path, max_width: int = MAX_WEB_WIDTH):
    """Re-encode to H.264/AAC in an MP4 container (browser-playable) and
    scale down large frames, using ffmpeg."""
    scale_filter = f"scale='min({max_width},iw)':-2"
    cmd = [
        "ffmpeg", "-y", "-i", str(src),
        "-vf", scale_filter,
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "26",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-an",
        str(dst),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed on {src}:\n{result.stderr[-2000:]}")


def build_zone_preview(video_path: str, zones, out_path: Path, max_width: int = 1000):
    """Save a single JPEG of the video's first frame with worker zones
    overlaid, so the demo can explain the zone without shipping the tool."""
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print(f"  ⚠️  Could not read a frame from {video_path} for zone preview")
        return

    h, w = frame.shape[:2]
    scale = min(max_width / w, 1.0)
    frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

    if zones:
        overlay = frame.copy()
        for zone in zones:
            pts = np.array([[int(x * scale), int(y * scale)] for x, y in zone], dtype=np.int32)
            cv2.fillPoly(overlay, [pts], (0, 0, 255))
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        for idx, zone in enumerate(zones, 1):
            pts = np.array([[int(x * scale), int(y * scale)] for x, y in zone], dtype=np.int32)
            cv2.polylines(frame, [pts], True, (0, 0, 255), 2)
            centroid = pts.mean(axis=0).astype(int)
            cv2.putText(frame, f"STAFF ZONE {idx}", tuple(centroid),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 3)
            cv2.putText(frame, f"STAFF ZONE {idx}", tuple(centroid),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imwrite(str(out_path), frame)


def main():
    if len(sys.argv) != 4:
        print("Usage: python build_demo_data.py <slug> <label> <original_video_path>")
        sys.exit(1)

    slug, label, original_video_path = sys.argv[1], sys.argv[2], sys.argv[3]

    analysis_path = Path("results/final_corrected_times.json")
    dwell_csv_path = Path("results/initial_tracking.csv")
    tracked_video_path = Path(Config.OUTPUT_VIDEO_PATH)

    if not analysis_path.exists():
        print(f"❌ {analysis_path} not found - run run_tracking.py first")
        sys.exit(1)

    out_dir = DEMO_DATA_DIR / slug
    segments_out_dir = out_dir / "segments"
    segments_out_dir.mkdir(parents=True, exist_ok=True)

    print(f"📦 Packaging demo data for '{slug}' ({label})")

    with open(analysis_path) as f:
        analysis = json.load(f)

    # Re-encode + copy each confusion segment clip, rewriting its path to be
    # relative and web-friendly.
    for i, correction in enumerate(analysis.get("corrections", [])):
        src = Path(correction["segment_path"])
        if not src.exists():
            print(f"  ⚠️  Missing segment clip: {src}")
            continue
        dst_name = f"segment_{i + 1}.mp4"
        dst = segments_out_dir / dst_name
        print(f"  🎬 Re-encoding segment {i + 1}: {src.name}")
        reencode_for_web(src, dst)
        correction["segment_path"] = f"segments/{dst_name}"

    with open(out_dir / "analysis.json", "w") as f:
        json.dump(analysis, f, indent=2)

    if dwell_csv_path.exists():
        shutil.copy(dwell_csv_path, out_dir / "dwell_times.csv")

    if tracked_video_path.exists():
        print(f"  🎬 Re-encoding tracked video...")
        reencode_for_web(tracked_video_path, out_dir / "tracked_video.mp4")
    else:
        print(f"  ⚠️  {tracked_video_path} not found")

    zones = Config.get_worker_zones()
    build_zone_preview(original_video_path, zones, out_dir / "zone_preview.jpg")

    # Update the manifest
    manifest_path = DEMO_DATA_DIR / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else []
    manifest = [m for m in manifest if m["slug"] != slug]
    manifest.append({
        "slug": slug,
        "label": label,
        "total_customers": len(analysis.get("final_times", {})),
        "confusions_detected": len(analysis.get("corrections", [])),
    })
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"✅ Demo data ready: {out_dir}/")


if __name__ == "__main__":
    main()
