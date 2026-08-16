"""
Streamlit webapp for ANAVID Queue Intelligence System
Complete integration of all services with dynamic UI and live footage display
"""

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import json
import os
import threading
import time
import subprocess
import sys
import platform
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import tempfile

# Import project modules
from config import Config
from queue_tracker import DwellTimeTracker
from tools.segment_extractor import extract_segment_with_context
from agent import create_segment_analyzer_agent, analyze_confusion_segment


# ==================== STREAMLIT PAGE CONFIG ====================
st.set_page_config(
    page_title="🏪 ANAVID Queue Intelligence",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS ====================
st.markdown("""
<style>
    /* Main container */
    .main {
        background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
        color: #ffffff;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(135deg, #0f0f0f 0%, #1a1a1a 100%);
    }
    
    /* Metric boxes */
    .metric-box {
        background: rgba(255, 222, 89, 0.1);
        border: 2px solid rgba(255, 222, 89, 0.4);
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        backdrop-filter: blur(10px);
    }
    
    /* Status indicators */
    .status-success {
        color: #ffde59;
    }
    
    .status-warning {
        color: #ff914d;
    }
    
    .status-error {
        color: #ff4444;
    }
    
    /* Log section */
    .log-section {
        background: rgba(0, 0, 0, 0.3);
        border-left: 4px solid #ffde59;
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
        font-family: 'Courier New', monospace;
        font-size: 0.9em;
    }
    
    h1, h2, h3 {
        background: linear-gradient(90deg, #ffde59, #ff914d);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* Video container */
    .video-container {
        border: 3px solid rgba(255, 222, 89, 0.4);
        border-radius: 10px;
        padding: 10px;
        background: rgba(0, 0, 0, 0.5);
    }
    
    /* Stats box */
    .stats-box {
        background: rgba(0, 0, 0, 0.7);
        padding: 15px;
        border-radius: 10px;
        border: 2px solid rgba(255, 145, 77, 0.4);
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(90deg, #ffde59, #ff914d);
        color: #000000;
        font-weight: bold;
        border-radius: 10px;
        border: none;
        padding: 0.5rem 1rem;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 4px 15px rgba(255, 222, 89, 0.6);
    }
</style>
""", unsafe_allow_html=True)

# ==================== SESSION STATE INITIALIZATION ====================
if 'tracker' not in st.session_state:
    st.session_state.tracker = None
if 'processing_logs' not in st.session_state:
    st.session_state.processing_logs = []
if 'confusion_results' not in st.session_state:
    st.session_state.confusion_results = []
if 'final_results' not in st.session_state:
    st.session_state.final_results = {}
if 'current_frame' not in st.session_state:
    st.session_state.current_frame = None
if 'video_cap' not in st.session_state:
    st.session_state.video_cap = None
if 'selected_video' not in st.session_state:
    st.session_state.selected_video = None
if 'last_results_check' not in st.session_state:
    st.session_state.last_results_check = 0
if 'auto_refresh_enabled' not in st.session_state:
    st.session_state.auto_refresh_enabled = False
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'current_stats' not in st.session_state:
    st.session_state.current_stats = {}
if 'latest_annotated_frame' not in st.session_state:
    st.session_state.latest_annotated_frame = None


# ==================== UTILITY FUNCTIONS ====================
def add_log(log_message: str, log_type: str = "info"):
    """Add message to processing logs with timestamp and styling"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    if log_type == "success":
        icon = "✅"
        style = "status-success"
    elif log_type == "error":
        icon = "❌"
        style = "status-error"
    elif log_type == "warning":
        icon = "⚠️"
        style = "status-warning"
    else:
        icon = "ℹ️"
        style = "info"
    
    log_entry = {
        'timestamp': timestamp,
        'message': log_message,
        'type': log_type,
        'icon': icon,
        'style': style
    }
    
    st.session_state.processing_logs.append(log_entry)
    print(f"[{timestamp}] {icon} {log_message}")


def display_logs():
    """Display all processing logs in aesthetic format"""
    if st.session_state.processing_logs:
        st.markdown("### 📋 Processing Logs")
        for log in st.session_state.processing_logs:
            if log['type'] == 'success':
                st.success(f"{log['icon']} [{log['timestamp']}] {log['message']}")
            elif log['type'] == 'error':
                st.error(f"{log['icon']} [{log['timestamp']}] {log['message']}")
            elif log['type'] == 'warning':
                st.warning(f"{log['icon']} [{log['timestamp']}] {log['message']}")
            else:
                st.info(f"{log['icon']} [{log['timestamp']}] {log['message']}")


def get_video_files() -> List[str]:
    """Get list of available videos from data/input directory"""
    input_dir = Path("data/input")
    input_dir.mkdir(parents=True, exist_ok=True)
    
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv']
    videos = []
    
    for ext in video_extensions:
        videos.extend([str(f) for f in input_dir.glob(f'*{ext}')])
    
    return sorted(videos) if videos else []


def get_output_files(file_type: str = "json", force_refresh: bool = False) -> List[str]:
    """Get list of available output files with optional cache bypass"""
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    if file_type == "json":
        files = sorted([str(f) for f in results_dir.glob("*.json")])
    elif file_type == "csv":
        files = sorted([str(f) for f in results_dir.glob("*.csv")])
    else:
        files = sorted([str(f) for f in results_dir.glob("*")])
    
    return files


def get_latest_result_file(file_type: str = "json") -> Optional[str]:
    """Get the most recently modified result file"""
    files = get_output_files(file_type, force_refresh=True)
    if not files:
        return None
    
    # Return the most recently modified file
    return max(files, key=lambda f: Path(f).stat().st_mtime)


def count_result_files() -> Dict[str, int]:
    """Count available result files by type"""
    return {
        'json': len(get_output_files('json', force_refresh=True)),
        'csv': len(get_output_files('csv', force_refresh=True)),
        'total': len(get_output_files('all', force_refresh=True))
    }


def open_video_capture(video_path: str):
    """Safely open video capture"""
    if st.session_state.video_cap:
        st.session_state.video_cap.release()
    st.session_state.video_cap = cv2.VideoCapture(video_path)
    return st.session_state.video_cap


def get_next_frame(cap):
    """Get next frame from video capture"""
    ret, frame = cap.read()
    if ret:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return None


def _reset_zone_editor(video_path: str):
    """Reset in-browser zone editor state for a (new) video."""
    st.session_state.zone_editor_video = video_path
    st.session_state.zone_editor_zones = []
    st.session_state.zone_editor_current_points = []
    st.session_state.zone_editor_last_click = None


def _draw_zones_on_frame(display_frame, zones, scale, color=(255, 0, 0), labels=False):
    """Draw filled+outlined zone polygons (given in original-resolution
    coords) onto a display-scaled frame. Returns the annotated frame."""
    if not zones:
        return display_frame

    overlay = display_frame.copy()
    for zone in zones:
        pts = np.array([[int(x * scale), int(y * scale)] for x, y in zone], dtype=np.int32)
        cv2.fillPoly(overlay, [pts], color)
    display_frame = cv2.addWeighted(display_frame, 0.7, overlay, 0.3, 0)

    for idx, zone in enumerate(zones, 1):
        pts = np.array([[int(x * scale), int(y * scale)] for x, y in zone], dtype=np.int32)
        cv2.polylines(display_frame, [pts], True, color, 2)
        if labels:
            centroid = pts.mean(axis=0).astype(int)
            cv2.putText(display_frame, f"ZONE {idx}", tuple(centroid),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 3)
            cv2.putText(display_frame, f"ZONE {idx}", tuple(centroid),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    return display_frame


def render_zone_preview(video_path: str, zones, max_display_width: int = 900):
    """Show the currently saved worker zones overlaid on the video's first
    frame, so it's obvious at a glance what's configured instead of reading
    raw pixel coordinates."""
    if not zones:
        st.warning("No worker zones configured")
        return

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        st.error(f"Could not read a frame from {video_path} to preview zones")
        return

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = frame.shape[:2]
    scale = min(max_display_width / orig_w, 1.0)
    display_frame = cv2.resize(frame, (int(orig_w * scale), int(orig_h * scale)))
    display_frame = _draw_zones_on_frame(display_frame, zones, scale, labels=True)

    st.image(
        Image.fromarray(display_frame),
        caption=f"{len(zones)} zone(s) currently saved to .env (shown on {Path(video_path).name})",
        use_container_width=True,
    )

    with st.expander("Raw zone coordinates"):
        for idx, zone in enumerate(zones, 1):
            st.write(f"**Zone {idx}:** {len(zone)} points — {zone}")


def render_zone_editor(video_path: str, max_display_width: int = 900):
    """
    In-browser worker-zone drawing tool: click points on the first frame to
    build a polygon, complete it, repeat for more zones, then save to .env.
    Mirrors workzone.py's ZoneDefiner but runs entirely inside Streamlit
    instead of opening a native OpenCV window.
    """
    from streamlit_image_coordinates import streamlit_image_coordinates

    if st.session_state.get('zone_editor_video') != video_path:
        _reset_zone_editor(video_path)

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        st.error(f"Could not read a frame from {video_path}")
        return

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = frame.shape[:2]
    scale = min(max_display_width / orig_w, 1.0)
    disp_w, disp_h = int(orig_w * scale), int(orig_h * scale)
    display_frame = cv2.resize(frame, (disp_w, disp_h))

    zones = st.session_state.zone_editor_zones
    current_points = st.session_state.zone_editor_current_points

    # Draw completed zones (in original-resolution coords, scaled for display)
    display_frame = _draw_zones_on_frame(display_frame, zones, scale)

    # Draw the zone currently being defined
    if current_points:
        disp_pts = [(int(x * scale), int(y * scale)) for x, y in current_points]
        for pt in disp_pts:
            cv2.circle(display_frame, pt, 6, (0, 255, 0), -1)
        if len(disp_pts) > 1:
            cv2.polylines(display_frame, [np.array(disp_pts, dtype=np.int32)], False, (0, 255, 0), 2)

    st.caption(
        f"Click to add a point to the current zone "
        f"({len(current_points)} point(s) so far, {len(zones)} zone(s) completed)."
    )

    click = streamlit_image_coordinates(
        Image.fromarray(display_frame), key=f"zone_click_{video_path}"
    )

    if click is not None and click != st.session_state.zone_editor_last_click:
        st.session_state.zone_editor_last_click = click
        orig_point = (click['x'] / scale, click['y'] / scale)
        st.session_state.zone_editor_current_points.append(orig_point)
        st.rerun()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("✅ Complete Zone", use_container_width=True,
                     disabled=len(current_points) < 3):
            st.session_state.zone_editor_zones.append(current_points.copy())
            st.session_state.zone_editor_current_points = []
            st.rerun()
    with col2:
        if st.button("↩️ Undo Point", use_container_width=True,
                     disabled=not current_points):
            st.session_state.zone_editor_current_points.pop()
            st.rerun()
    with col3:
        if st.button("🔄 Reset All", use_container_width=True):
            _reset_zone_editor(video_path)
            st.rerun()
    with col4:
        if st.button("💾 Save Zones to .env", use_container_width=True,
                     disabled=not zones, type="primary"):
            Config.update_worker_zones(zones)
            add_log(f"Saved {len(zones)} zone(s) to .env", "success")
            st.success(f"✅ Saved {len(zones)} zone(s)!")


def display_metrics(tracker: DwellTimeTracker):
    """Display key metrics in a beautiful format"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "👥 Customers Tracked",
            len(tracker.person_first_seen),
            delta=None,
            delta_color="off"
        )
    
    with col2:
        st.metric(
            "⚠️ Confusions Detected",
            len(tracker.confusion_events),
            delta=None,
            delta_color="off"
        )
    
    with col3:
        if tracker.person_first_seen:
            avg_time = np.mean([
                tracker.person_total_time.get(pid, 0)
                for pid in tracker.person_first_seen.keys()
            ])
            st.metric(
                "⏱️ Avg Dwell Time",
                f"{avg_time:.1f}s",
                delta=None,
                delta_color="off"
            )
    
    with col4:
        st.metric(
            "💾 Results Saved",
            len(get_output_files()),
            delta=None,
            delta_color="off"
        )


def load_json_result(file_path: str) -> Dict:
    """Load JSON result file"""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        add_log(f"Error loading {file_path}: {str(e)}", "error")
        return {}


def load_csv_result(file_path: str) -> pd.DataFrame:
    """Load CSV result file"""
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        add_log(f"Error loading {file_path}: {str(e)}", "error")
        return pd.DataFrame()


# ==================== SIDEBAR CONFIGURATION ====================
st.sidebar.markdown("## ⚙️ Configuration")

config_section = st.sidebar.expander("📋 System Configuration", expanded=True)
with config_section:
    st.write("**Current Configuration:**")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"🎯 YOLO Model: `{Config.YOLO_MODEL}`")
        st.write(f"📊 FPS Limit: `{Config.FPS_LIMIT}`")
    
    with col2:
        st.write(f"⏱️ Disappear Threshold: `{Config.DISAPPEAR_THRESHOLD}s`")
        
        worker_zones = Config.get_worker_zones()
        st.write(f"🏢 Worker Zones: `{len(worker_zones)}`")
    
    if st.button("🔄 Reload Configuration"):
        st.rerun()


# ==================== MAIN APP ====================
st.title("🏪 ANAVID Queue Intelligence System")
st.markdown("**Real-time Customer Dwell Time Tracking & Confusion Detection**")

# Create tabs for different services
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🎥 Live Tracking",
    "📊 Results Analysis",
    "🎬 Confusion Segments",
    "📈 Analytics Dashboard",
    "⚙️ Zone Management",
    "❓ Help"
])

# ==================== TAB 1: LIVE TRACKING ====================
with tab1:
    st.header("🎥 Live YOLO Tracking with Confusion Detection")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("Video Selection")
        video_files = get_video_files()
        
        if video_files:
            selected_video = st.selectbox(
                "Select video to process:",
                video_files,
                key="video_select"
            )
        else:
            st.warning("No video files found in `data/input/` directory")
            selected_video = None
    
    with col2:
        st.write("") # Spacing
        if st.button("🔄 Refresh Videos", use_container_width=True):
            st.rerun()
    
    if selected_video:
        # Video preview and frame display
        st.markdown("---")
        col_video1, col_video2 = st.columns([2, 1])
        
        with col_video1:
            st.markdown("### 📹 Video Preview")
            
            cap = cv2.VideoCapture(selected_video)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                st.image(
                    frame_rgb,
                    use_container_width=True,
                    caption="📸 First frame of selected video"
                )
            else:
                st.error("Could not read video file")
            cap.release()
        
        with col_video2:
            st.markdown("### ℹ️ Video Information")
            cap = cv2.VideoCapture(selected_video)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0
            cap.release()
            
            st.metric("⏱️ Duration", f"{duration:.1f}s")
            st.metric("📐 Resolution", f"{width}x{height}")
            st.metric("🎬 FPS", f"{fps:.1f}")
            st.metric("🎞️ Total Frames", f"{frame_count:,}")
            
            # File size
            try:
                file_size = os.path.getsize(selected_video) / (1024 * 1024)  # MB
                st.metric("💾 File Size", f"{file_size:.1f} MB")
            except:
                pass
        
        # Processing settings
        st.markdown("---")
        st.markdown("### ⚙️ Processing Settings")
        
        col_settings1, col_settings2 = st.columns(2)
        
        with col_settings1:
            fps_limit = st.slider(
                "🎬 FPS Limit",
                min_value=1,
                max_value=30,
                value=Config.FPS_LIMIT,
                step=1,
                help="Lower values = faster processing but less smooth tracking"
            )
        
        with col_settings2:
            disappear_threshold = st.slider(
                "⏱️ Disappear Threshold (seconds)",
                min_value=1.0,
                max_value=30.0,
                value=Config.DISAPPEAR_THRESHOLD,
                step=0.5,
                help="Seconds before person is considered to have left"
            )
        
        st.info("💡 **Tip**: Use FPS Limit 5-10 for quick testing, 15-30 for accuracy. Disappear Threshold: 5-10s for busy areas, 10-15s for sparse areas.")
        
        # GPU Status Check
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                st.success(f"✅ **GPU Available**: {gpu_name} ({gpu_memory:.1f} GB) - Processing will use CUDA")
            else:
                st.warning("⚠️ **No GPU Detected** - Processing will use CPU (much slower)")
        except ImportError:
            st.info("ℹ️ PyTorch not available - cannot check GPU status")
        
        # Processing mode selection
        st.markdown("---")
        processing_mode = st.radio(
            "🎯 Processing Mode:",
            ["🖥️ Terminal Window (Live OpenCV Preview)", "🌐 Web Interface Only"],
            help="Terminal mode opens a new window with live tracking visualization. Web mode processes in background."
        )
        
        # Start processing button
        if st.button("🚀 START PROCESSING", key="start_tracking", use_container_width=True):
            st.session_state.processing = True
            add_log("Initializing queue tracker...", "info")
            
            try:
                if "Terminal Window" in processing_mode:
                    # Launch in terminal window - FULL AI ANALYSIS
                    add_log("Launching complete AI analysis system in new terminal window...", "info")
                    
                    # Get Python executable - prefer virtual environment with CUDA
                    python_exe = sys.executable
                    
                    # Check if we're in a venv, if not try to use anavid_py311 venv
                    venv_py311 = os.path.join(os.getcwd(), "anavid_py311", "Scripts", "python.exe")
                    venv_anavid = os.path.join(os.getcwd(), "anavid", "Scripts", "python.exe")
                    
                    if os.path.exists(venv_py311):
                        python_exe = venv_py311
                        add_log(f"✅ Using virtual environment: anavid_py311 (has CUDA support)", "success")
                    elif os.path.exists(venv_anavid):
                        python_exe = venv_anavid
                        add_log(f"✅ Using virtual environment: anavid (has CUDA support)", "success")
                    else:
                        # Check if current Python has CUDA
                        try:
                            import torch
                            if not torch.cuda.is_available():
                                add_log("⚠️ System Python doesn't have CUDA - consider using virtual environment", "warning")
                        except:
                            pass
                    
                    script_path = os.path.join(os.getcwd(), "run_tracking.py")
                    
                    # Build command
                    cmd = [
                        python_exe,
                        script_path,
                        selected_video,
                        str(fps_limit),
                        str(disappear_threshold)
                    ]
                    
                    # Launch in new terminal window based on OS
                    if platform.system() == "Windows":
                        # Windows: Use start command to open new terminal
                        # Build command string for Windows cmd
                        # Use absolute paths and proper escaping
                        abs_script = os.path.abspath(script_path)
                        abs_video = os.path.abspath(selected_video)
                        
                        # Create command that works in Windows cmd
                        cmd_str = f'{python_exe} "{abs_script}" "{abs_video}" {fps_limit} {disappear_threshold}'
                        
                        # Use start to open new cmd window with title
                        # /k keeps window open after completion
                        subprocess.Popen(
                            f'start "Queue Intelligence - Full AI Analysis" cmd /k "{cmd_str}"',
                            shell=True
                        )
                        add_log("✅ Terminal window opened! Full AI analysis will run in the new window.", "success")
                        add_log("📺 Look for the OpenCV window showing live YOLO tracking", "info")
                        add_log("🤖 AI agent will analyze confusion events after tracking completes", "info")
                        st.success("🚀 **Terminal window opened!** Full AI analysis pipeline starting...")
                        st.info("💡 The OpenCV window will show live tracking. AI analysis logs will appear in the terminal.")
                        st.session_state.processing = False
                        
                    elif platform.system() == "Darwin":  # macOS
                        # macOS: Use open -a Terminal
                        script_content = f"""#!/bin/bash
cd "{os.getcwd()}"
{python_exe} "{script_path}" "{selected_video}" {fps_limit} {disappear_threshold}
"""
                        temp_script = tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False)
                        temp_script.write(script_content)
                        temp_script.close()
                        os.chmod(temp_script.name, 0o755)
                        
                        subprocess.Popen([
                            "open", "-a", "Terminal", temp_script.name
                        ])
                        add_log("✅ Terminal window opened!", "success")
                        st.success("🚀 **Terminal window opened!** Check the new window for live tracking.")
                        st.session_state.processing = False
                        
                    else:  # Linux
                        # Linux: Use xterm or gnome-terminal
                        terminal_cmd = "xterm" if os.system("which xterm > /dev/null 2>&1") == 0 else "gnome-terminal"
                        subprocess.Popen([
                            terminal_cmd, "-e",
                            f"{python_exe} {script_path} {selected_video} {fps_limit} {disappear_threshold}"
                        ])
                        add_log("✅ Terminal window opened!", "success")
                        st.success("🚀 **Terminal window opened!** Check the new window for live tracking.")
                        st.session_state.processing = False
                    
                    # Record when this run started, so the results panel below
                    # (rendered outside this button's if-block, further down)
                    # can tell fresh output apart from a previous run's leftovers.
                    st.session_state.terminal_run_started_at = time.time()
                    st.session_state.terminal_run_pending = True

                    st.info("⏳ Processing in terminal window... scroll down for live status.")

                else:
                    # Web interface mode (original behavior)
                    worker_zones = Config.get_worker_zones()
                    
                    # Check GPU availability before creating tracker
                    import torch
                    if torch.cuda.is_available():
                        gpu_name = torch.cuda.get_device_name(0)
                        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                        add_log(f"✅ CUDA GPU detected: {gpu_name} ({gpu_memory:.1f} GB)", "success")
                        st.success(f"🚀 **GPU Acceleration Enabled**: {gpu_name}")
                    else:
                        add_log("⚠️ CUDA not available - will use CPU (much slower)", "warning")
                        st.warning("⚠️ **No GPU detected** - Processing will be slower on CPU")
                    
                    # Create tracker
                    tracker = DwellTimeTracker(
                        model_path=Config.YOLO_MODEL,
                        disappear_threshold=disappear_threshold,
                        fps_limit=fps_limit,
                        exclude_zones=worker_zones
                    )
                    
                    # Verify device after initialization
                    if hasattr(tracker.model, 'device'):
                        actual_device = tracker.model.device
                        if 'cuda' in str(actual_device):
                            add_log(f"✅ Model loaded on GPU: {actual_device}", "success")
                        else:
                            add_log(f"⚠️ Model on CPU: {actual_device}", "warning")
                    
                    add_log(f"Tracker initialized | Zones: {len(worker_zones)}", "success")
                    
                    # Create UI placeholders
                    col_video, col_stats = st.columns([3, 1])
                    
                    with col_video:
                        st.markdown("### 🎥 Live YOLO Tracking")
                        frame_placeholder = st.empty()
                    
                    with col_stats:
                        st.markdown("### 📊 Real-time Stats")
                        stats_placeholder = st.empty()
                        progress_placeholder = st.empty()
                    
                    # Progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Process video (OPTIMIZED - no callback overhead for maximum speed)
                    add_log("Processing video with YOLO tracking...", "info")
                    add_log("Note: Tracking visualization will appear after processing completes", "info")
                    add_log("Performance: Frame collection optimized for speed", "info")
                    
                    # Process video WITHOUT callback for maximum speed
                    # Callback adds significant overhead - we'll show final result instead
                    tracker.process_video(
                        selected_video,
                        output_path=Config.OUTPUT_VIDEO_PATH,
                        show_preview=False,  # Don't show OpenCV window
                        frame_callback=None  # DISABLED for performance - use terminal mode for live preview
                    )
                    
                    # After processing, load and display the final frame from output video
                    add_log("Loading final tracking result...", "info")
                    try:
                        cap = cv2.VideoCapture(Config.OUTPUT_VIDEO_PATH)
                        if cap.isOpened():
                            # Get last frame
                            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                            if total_frames > 0:
                                cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
                                ret, final_frame = cap.read()
                                if ret:
                                    final_frame_rgb = cv2.cvtColor(final_frame, cv2.COLOR_BGR2RGB)
                                    st.session_state.latest_annotated_frame = final_frame_rgb
                        cap.release()
                    except Exception as e:
                        add_log(f"Could not load final frame: {e}", "warning")
                    
                    add_log(f"Tracked {len(tracker.person_first_seen)} customers", "success")
                    add_log(f"Detected {len(tracker.confusion_events)} confusion events", "warning" if tracker.confusion_events else "success")
                    
                    # Save results
                    os.makedirs("results", exist_ok=True)
                    tracker.save_results("results/initial_tracking.csv")
                    tracker.save_confusion_report("results/confusion_events.json")
                    
                    add_log("Results saved to results/ directory", "success")
                    
                    # Store tracker in session state
                    st.session_state.tracker = tracker
                    st.session_state.selected_video = selected_video
                    st.session_state.processing = False
                    
                    # Display final annotated frame with tracking
                    if st.session_state.latest_annotated_frame is not None:
                        frame_placeholder.image(
                            st.session_state.latest_annotated_frame,
                            caption="🎯 YOLO Tracking Result - Final Frame (with bounding boxes, IDs, and dwell times)",
                            use_container_width=True,
                            channels="RGB"
                        )
                        
                        # Display final stats
                        final_stats = st.session_state.current_stats
                        stats_html = f"""
                        <div class="stats-box" style="color: white;">
                            <h4 style="color: #00ff41; margin-bottom: 15px;">📊 Final Statistics</h4>
                            <p><strong>👥 Customers Visible:</strong> {final_stats.get('customers_visible', 0)}</p>
                            <p><strong>👤 Total Tracked:</strong> {final_stats.get('total_tracked', 0)}</p>
                            <p><strong>✅ Completed:</strong> {final_stats.get('completed', 0)}</p>
                            <p><strong>⚠️ Confusions:</strong> {final_stats.get('confusions', 0)}</p>
                            <p><strong>👷 Workers:</strong> {final_stats.get('workers', 0)}</p>
                            <p><strong>⏱️ Video Time:</strong> {final_stats.get('time', 0):.1f}s</p>
                        </div>
                        """
                        stats_placeholder.markdown(stats_html, unsafe_allow_html=True)
                    
                    # Note about performance optimization
                    st.info("💡 **Performance Note**: For live preview during processing, use 'Terminal Window' mode. Web mode is optimized for speed and shows results after completion.")
                    
                    progress_bar.progress(1.0)
                    status_text.text("✅ Processing Complete! YOLO tracking visualization shown above.")
                    
                    # Show output video if it exists
                    if os.path.exists(Config.OUTPUT_VIDEO_PATH):
                        st.markdown("---")
                        st.markdown("### 🎬 Complete Tracked Video")
                        st.info("Full video with YOLO tracking annotations saved. You can download it or play it below.")
                        
                        video_file = open(Config.OUTPUT_VIDEO_PATH, 'rb')
                        video_bytes = video_file.read()
                        st.video(video_bytes)
                        video_file.close()
                        
                        st.download_button(
                            label="📥 Download Tracked Video",
                            data=video_bytes,
                            file_name=os.path.basename(Config.OUTPUT_VIDEO_PATH),
                            mime="video/mp4"
                        )
                    
                    # Display metrics
                    st.markdown("---")
                    display_metrics(tracker)
                    
                    # Display tracking results table
                    st.markdown("### 📊 Tracking Results")
                    
                    results_data = []
                    for person_id, track_data in tracker.person_tracks.items():
                        dwell_time = track_data['last_seen'] - track_data['first_seen']
                        results_data.append({
                            'Customer ID': person_id,
                            'Dwell Time (s)': f"{dwell_time:.2f}",
                            'Entry Time (s)': f"{track_data['first_seen']:.2f}",
                            'Exit Time (s)': f"{track_data['last_seen']:.2f}",
                            'Total Time': f"{track_data['total_time']:.2f}s"
                        })
                    
                    if results_data:
                        df_results = pd.DataFrame(results_data)
                        st.dataframe(df_results, width='stretch', hide_index=True)
                    
                    # Show logs
                    display_logs()
                    
            except Exception as e:
                st.session_state.processing = False
                add_log(f"Error during processing: {str(e)}", "error")
                st.error(f"Processing failed: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
                display_logs()

    # Persistent status panel for a Terminal Window run: rendered on every
    # rerun of this tab (not just the click that launched it), so "Check for
    # Results" actually does something. Result files only count as belonging
    # to THIS run if their mtime is after the run's start time - otherwise
    # they're a previous run's leftovers, which used to be shown as if they
    # were fresh (the bug this replaced).
    if st.session_state.get('terminal_run_pending'):
        st.markdown("---")
        st.markdown("### 🖥️ Terminal Window Run Status")

        run_started_at = st.session_state.get('terminal_run_started_at', 0)

        col_check, col_dismiss = st.columns([3, 1])
        with col_check:
            if st.button("🔄 Check for Results", key="check_terminal_results", use_container_width=True):
                st.rerun()
        with col_dismiss:
            if st.button("✖️ Dismiss", key="dismiss_terminal_results", use_container_width=True):
                st.session_state.terminal_run_pending = False
                st.rerun()

        def _fresh_result(path: str) -> bool:
            return os.path.exists(path) and os.path.getmtime(path) >= run_started_at

        results_csv = "results/initial_tracking.csv"
        results_json = "results/confusion_events.json"
        ai_results_json = "results/final_corrected_times.json"

        if _fresh_result(results_csv):
            st.success("✅ **Initial tracking results found for this run!**")
            try:
                df = pd.read_csv(results_csv)
                if not df.empty:
                    st.markdown("#### 📊 Initial Tracking Results")
                    st.dataframe(df, width='stretch', hide_index=True)
                    if 'total_time' in df.columns:
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Customers", len(df))
                        with col2:
                            st.metric("Avg Dwell Time", f"{df['total_time'].mean():.1f}s")
                        with col3:
                            st.metric("Max Dwell Time", f"{df['total_time'].max():.1f}s")
            except Exception as e:
                st.error(f"Error loading results: {e}")
        else:
            st.info(
                "⏳ **Waiting for results from this run...** Processing is "
                "still running in the terminal window (or hasn't produced "
                "output yet). Click 'Check for Results' to refresh."
            )

        if _fresh_result(ai_results_json):
            st.success("🤖 **AI Analysis Complete!** Full analytics available.")

            with open(ai_results_json, 'r') as f:
                ai_data = json.load(f)

            st.markdown("#### 🤖 AI Analysis Summary")

            if 'ai_analysis_log' in ai_data and len(ai_data['ai_analysis_log']) > 0:
                st.markdown(f"**Total Confusions Analyzed:** {len(ai_data['ai_analysis_log'])}")

                ai_logs = []
                for log in ai_data['ai_analysis_log']:
                    ai_logs.append({
                        'Confusion #': log['confusion_id'],
                        'Type': log['event_type'],
                        'Person ID': log['person_id'],
                        'Time (s)': f"{log['timestamp']:.1f}",
                        'AI Decision': log['ai_decision'].upper(),
                        'Confidence': f"{log['ai_confidence']:.1%}",
                        'Reasoning Preview': log['ai_reasoning'][:100] + "..." if len(log['ai_reasoning']) > 100 else log['ai_reasoning']
                    })

                df_ai = pd.DataFrame(ai_logs)
                st.dataframe(df_ai, width='stretch', hide_index=True)

                with st.expander("📋 View Full AI Analysis Logs", expanded=False):
                    st.json(ai_data['ai_analysis_log'])

            if 'final_times' in ai_data:
                st.markdown("#### 📊 Final Corrected Times (AI-Adjusted)")
                corrected_data = []
                for person_id, data in ai_data['final_times'].items():
                    corrected_data.append({
                        'Customer ID': person_id,
                        'Original Time (s)': f"{data['original_time']:.2f}",
                        'Corrected Time (s)': f"{data['corrected_time']:.2f}",
                        'Difference (s)': f"{data['corrected_time'] - data['original_time']:+.2f}",
                        'AI Corrections': len(data['corrections_applied'])
                    })

                if corrected_data:
                    df_corrected = pd.DataFrame(corrected_data)
                    st.dataframe(df_corrected, width='stretch', hide_index=True)

            st.download_button(
                label="📥 Download Full AI Analysis Report",
                data=json.dumps(ai_data, indent=2),
                file_name="ai_analysis_report.json",
                mime="application/json"
            )

        if _fresh_result(results_json):
            st.success("✅ Confusion events detected!")
            if st.button("📄 View Confusion Events", key="view_confusion"):
                with open(results_json, 'r') as f:
                    confusion_data = json.load(f)
                st.json(confusion_data)

        display_logs()


# ==================== TAB 2: RESULTS ANALYSIS ====================
with tab2:
    st.header("📊 Results Analysis")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.subheader("Load Results Files")
    
    with col2:
        if st.button("🔄 Refresh", use_container_width=True, key="refresh_results"):
            st.rerun()
    
    with col3:
        auto_refresh = st.checkbox(
            "🔁 Auto-refresh",
            value=st.session_state.auto_refresh_enabled,
            help="Reloads the whole page every 2s to pick up new result files. "
                 "Streamlit reruns every tab's code on each refresh, so turn "
                 "this off before using other tabs (e.g. Zone Management).",
        )
        st.session_state.auto_refresh_enabled = auto_refresh

    # Auto-refresh every 2 seconds if enabled (reloads the whole app - see checkbox help text)
    if st.session_state.auto_refresh_enabled:
        import time
        time.sleep(2)
        st.rerun()
    
    # Display file counts
    file_counts = count_result_files()
    st.info(f"📁 Available Results: {file_counts['json']} JSON | {file_counts['csv']} CSV | {file_counts['total']} Total")
    
    # ===== JSON Results =====
    st.markdown("### 📄 JSON Results")
    json_files = get_output_files("json", force_refresh=True)
    
    if json_files:
        # Show latest file first
        latest_json = get_latest_result_file("json")
        default_idx = json_files.index(latest_json) if latest_json in json_files else 0
        
        selected_json = st.selectbox(
            "Select JSON result:",
            json_files,
            index=default_idx,
            key="json_select"
        )
        
        col_load, col_info = st.columns([3, 2])
        
        with col_load:
            if st.button("📂 Load JSON", key="load_json", use_container_width=True):
                data = load_json_result(selected_json)
                if data:
                    # Store in session
                    st.session_state.current_json = data
                    add_log(f"Loaded {Path(selected_json).name}", "success")
        
        with col_info:
            file_size = Path(selected_json).stat().st_size / 1024
            st.metric("File Size", f"{file_size:.1f} KB", label_visibility="collapsed")
        
        # Display JSON if loaded
        if 'current_json' in st.session_state:
            st.json(st.session_state.current_json)
            
            # Auto-expand specific sections
            if 'final_times' in st.session_state.current_json:
                st.markdown("#### 📊 Customer Times Summary")
                final_times = st.session_state.current_json['final_times']
                
                times_data = []
                for cust_id, data in final_times.items():
                    times_data.append({
                        'Customer ID': cust_id,
                        'Original Time (s)': f"{data['original_time']:.1f}",
                        'Corrected Time (s)': f"{data['corrected_time']:.1f}",
                        'Corrections': len(data.get('corrections_applied', []))
                    })
                
                if times_data:
                    df_times = pd.DataFrame(times_data)
                    st.dataframe(df_times, use_container_width=True, hide_index=True)
    else:
        st.info("📁 No JSON results found yet. Run tracking to generate results.")
    
    st.markdown("---")
    
    # ===== CSV Results =====
    st.markdown("### 📊 CSV Results")
    csv_files = get_output_files("csv", force_refresh=True)
    
    if csv_files:
        # Show latest file first
        latest_csv = get_latest_result_file("csv")
        default_csv_idx = csv_files.index(latest_csv) if latest_csv in csv_files else 0
        
        selected_csv = st.selectbox(
            "Select CSV result:",
            csv_files,
            index=default_csv_idx,
            key="csv_select"
        )
        
        col_load, col_info = st.columns([3, 2])
        
        with col_load:
            if st.button("📂 Load CSV", key="load_csv", use_container_width=True):
                df = load_csv_result(selected_csv)
                if not df.empty:
                    st.session_state.current_csv = df
                    add_log(f"Loaded {Path(selected_csv).name}", "success")
        
        with col_info:
            file_size = Path(selected_csv).stat().st_size / 1024
            st.metric("File Size", f"{file_size:.1f} KB", label_visibility="collapsed")
        
        # Display CSV if loaded
        if 'current_csv' in st.session_state:
            df = st.session_state.current_csv
            st.dataframe(df, use_container_width=True)
            
            # Display statistics
            st.markdown("#### 📈 Statistics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Records", len(df))
            
            with col2:
                if 'total_time' in df.columns:
                    avg_time = df['total_time'].mean()
                    st.metric("Avg Dwell Time", f"{avg_time:.1f}s")
            
            with col3:
                if 'total_time' in df.columns:
                    max_time = df['total_time'].max()
                    st.metric("Max Dwell Time", f"{max_time:.1f}s")
            
            with col4:
                if 'total_time' in df.columns:
                    med_time = df['total_time'].median()
                    st.metric("Median Time", f"{med_time:.1f}s")
    else:
        st.info("📁 No CSV results found yet. Run tracking to generate results.")
    
    st.markdown("---")
    display_logs()


# ==================== TAB 3: CONFUSION SEGMENTS ====================
with tab3:
    st.header("🎬 Confusion Segment Analysis")
    
    if st.session_state.tracker and st.session_state.tracker.confusion_events:
        st.info(f"Found {len(st.session_state.tracker.confusion_events)} confusion events")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            selected_confusion_idx = st.slider(
                "Select confusion event:",
                0,
                len(st.session_state.tracker.confusion_events) - 1,
                0
            )
        
        with col2:
            if st.button("🤖 Analyze with AI Agent", use_container_width=True):
                confusion = st.session_state.tracker.confusion_events[selected_confusion_idx]
                
                add_log(f"Analyzing confusion: {confusion.event_type}", "info")
                
                try:
                    add_log("Extracting segment context...", "info")
                    
                    segment_context = extract_segment_with_context(
                        video_path=st.session_state.selected_video,
                        confusion_event=confusion.to_dict(),
                        person_tracks=st.session_state.tracker.person_tracks,
                        padding_seconds=5.0
                    )
                    
                    add_log("Creating AI agent...", "info")
                    agent = create_segment_analyzer_agent()
                    
                    add_log("Analyzing with AI vision...", "info")
                    correction = analyze_confusion_segment(segment_context, agent)
                    
                    # Store results
                    st.session_state.confusion_results.append({
                        'confusion_event': confusion.to_dict(),
                        'segment_context': segment_context,
                        'agent_correction': correction
                    })
                    
                    add_log(f"Analysis complete | Decision: {correction['decision']}", "success")
                    
                except Exception as e:
                    add_log(f"Analysis failed: {str(e)}", "error")
        
        # Display confusion details
        confusion = st.session_state.tracker.confusion_events[selected_confusion_idx]
        
        st.markdown("### 🔍 Confusion Event Details")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Event Type", confusion.event_type.upper())
        
        with col2:
            st.metric("Person ID", confusion.person_id)
        
        with col3:
            st.metric("Timestamp", f"{confusion.timestamp:.2f}s")
        
        st.json(confusion.context)
        
        # Display analysis results if available
        if st.session_state.confusion_results:
            st.markdown("---")
            st.markdown("### 🤖 AI Analysis Results")
            
            for idx, result in enumerate(st.session_state.confusion_results):
                with st.expander(f"Analysis {idx + 1}: {result['agent_correction']['confusion_type']}", expanded=True):
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.metric("Decision", result['agent_correction']['decision'].upper())
                        st.metric("Confidence", f"{result['agent_correction']['confidence']:.2%}")
                    
                    with col2:
                        st.write("**Reasoning:**")
                        st.write(result['agent_correction']['reasoning'])
                    
                    st.write("**Corrected Times:**")
                    st.json(result['agent_correction']['corrected_times'])
    
    else:
        st.warning("No tracker or confusion events. Run tracking first in the 'Live Tracking' tab.")
    
    st.markdown("---")
    display_logs()


# ==================== TAB 4: ANALYTICS DASHBOARD ====================
with tab4:
    st.header("📈 Analytics Dashboard")
    
    # Try to load latest results first
    latest_json = get_latest_result_file("json")
    tracker_data = None
    
    if latest_json:
        try:
            with open(latest_json, 'r') as f:
                results_data = json.load(f)
                if 'final_times' in results_data:
                    tracker_data = results_data
                    st.success(f"📂 Loaded latest results: {Path(latest_json).name}")
        except Exception as e:
            st.warning(f"Could not load JSON: {e}")
    
    # Fallback to session tracker
    if not tracker_data and st.session_state.tracker and st.session_state.tracker.person_tracks:
        tracker = st.session_state.tracker
        
        # Prepare data for visualization
        dwell_times = []
        customer_ids = []
        
        for person_id, track_data in tracker.person_tracks.items():
            dwell_time = track_data['last_seen'] - track_data['first_seen']
            dwell_times.append(dwell_time)
            customer_ids.append(f"Customer {person_id}")
    elif tracker_data:
        # Extract data from JSON results
        final_times = tracker_data.get('final_times', {})
        dwell_times = []
        customer_ids = []
        
        for cust_id, data in final_times.items():
            dwell_times.append(data.get('corrected_time', data.get('original_time', 0)))
            customer_ids.append(f"Customer {cust_id}")
    else:
        dwell_times = []
        customer_ids = []
    
    if dwell_times:
        # Create visualizations
        col1, col2 = st.columns(2)
        
        # Dwell Time Distribution
        with col1:
            fig_dist = px.histogram(
                x=dwell_times,
                nbins=20,
                labels={'x': 'Dwell Time (seconds)', 'y': 'Count'},
                title='Dwell Time Distribution'
            )
            fig_dist.update_traces(marker_color='rgb(0, 255, 65)')
            st.plotly_chart(fig_dist, use_container_width=True)
        
        # Top Customers by Dwell Time
        with col2:
            top_n = min(10, len(customer_ids))
            sorted_data = sorted(
                zip(customer_ids, dwell_times),
                key=lambda x: x[1],
                reverse=True
            )[:top_n]
            
            customers_top, times_top = zip(*sorted_data) if sorted_data else ([], [])
            
            fig_bar = px.bar(
                x=times_top,
                y=customers_top,
                orientation='h',
                labels={'x': 'Dwell Time (s)', 'y': 'Customer'},
                title=f'Top {top_n} Customers by Dwell Time'
            )
            fig_bar.update_traces(marker_color='rgb(0, 212, 255)')
            st.plotly_chart(fig_bar, use_container_width=True)
        
        # Statistics
        st.markdown("---")
        st.markdown("### 📊 Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("👥 Total Customers", len(dwell_times))
        
        with col2:
            st.metric("⏱️ Avg Dwell Time", f"{np.mean(dwell_times):.1f}s")
        
        with col3:
            st.metric("📊 Median Time", f"{np.median(dwell_times):.1f}s")
        
        with col4:
            st.metric("⏳ Max Time", f"{np.max(dwell_times):.1f}s")
        
        # Show correction summary if available
        if tracker_data and 'ai_analysis_log' in tracker_data:
            st.markdown("---")
            st.markdown("### 🤖 AI Corrections Applied")
            
            ai_log = tracker_data['ai_analysis_log']
            corrections_count = len(ai_log)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("🔍 Confusions Detected", corrections_count)
            
            with col2:
                merged = sum(1 for item in ai_log if item.get('ai_decision') == 'MERGE')
                st.metric("✅ Merged", merged)
            
            with col3:
                avg_confidence = np.mean([item.get('ai_confidence', 0) for item in ai_log])
                st.metric("🎯 Avg Confidence", f"{avg_confidence:.0%}")
        
        st.markdown("---")
    
    else:
        st.warning("⚠️ No tracking data available. Run tracking first in the 'Live Tracking' tab.")
    
    st.markdown("---")
    display_logs()


# ==================== TAB 5: ZONE MANAGEMENT ====================
with tab5:
    st.header("⚙️ Worker Zone Management")
    
    st.info("Define worker/staff exclusion zones that won't be tracked as customers")

    video_files = get_video_files()
    if not video_files:
        st.error("No videos found in data/input/")
    else:
        selected_video_for_zones = st.selectbox(
            "Video:",
            video_files,
            key="zone_video_select",
            help="Zones are saved in pixel coordinates, so pick the video "
                 "whose camera framing they should match.",
        )

        st.markdown("### 📍 Current Worker Zones")
        render_zone_preview(selected_video_for_zones, Config.get_worker_zones())

        st.markdown("---")

        # Zone definition - drawn directly in the browser on the video's first frame
        st.markdown("### 🎯 Define New Zones")
        render_zone_editor(selected_video_for_zones)

    st.markdown("---")
    st.caption(
        "Prefer a native desktop window instead? Run "
        "`python workzone.py <video_path>` from a terminal — same tool, "
        "click-to-draw with mouse, saves to the same .env file."
    )

    st.markdown("---")
    
    # Manual zone input
    st.markdown("### 📝 Manual Zone Input (JSON)")
    
    st.write("Enter zones as JSON array of polygons:")
    
    example_zones = [
        [[100, 100], [300, 100], [300, 300], [100, 300]],
        [[400, 200], [600, 200], [600, 400], [400, 400]]
    ]
    
    zones_json_input = st.text_area(
        "Zones JSON:",
        value=json.dumps(example_zones, indent=2),
        height=200,
        help="Format: [[[x1, y1], [x2, y2], ...], ...]"
    )
    
    if st.button("💾 Save Zones", use_container_width=True):
        try:
            zones = json.loads(zones_json_input)
            # Convert to list of tuples
            zones_tuples = [
                [tuple(point) for point in zone]
                for zone in zones
            ]
            Config.update_worker_zones(zones_tuples)
            add_log(f"Saved {len(zones_tuples)} zones to .env", "success")
            st.success("✅ Zones saved!")
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON: {str(e)}")
    
    st.markdown("---")
    display_logs()


# ==================== TAB 6: HELP ====================
with tab6:
    st.header("❓ Help & Documentation")
    
    st.markdown("""
    ### 🚀 Getting Started
    
    1. **Live Tracking Tab**: Upload or select a video to process with YOLO object detection
    2. **Results Analysis**: View and analyze saved tracking results
    3. **Confusion Segments**: Review detected confusion events and get AI-powered corrections
    4. **Analytics Dashboard**: Visualize dwell times and customer patterns
    5. **Zone Management**: Define worker zones to exclude from tracking
    
    ---
    
    ### 📁 Data Structure
    
    - **Input Videos**: `data/input/` - Place your video files here
    - **Output Data**: `data/output/` - Processed videos and segments
    - **Results**: `results/` - JSON and CSV analysis files
    
    ---
    
    ### 🔑 Key Features
    
    - **YOLO v8 Detection**: Real-time person detection and tracking
    - **Dwell Time Calculation**: Accurate measurement of time spent per customer
    - **Confusion Detection**: Identifies when tracking gets confused (ID switches, occlusions)
    - **AI Analysis**: Uses Google ADK agent to analyze confusion segments
    - **Worker Zone Exclusion**: Automatically excludes worker areas from tracking
    
    ---
    
    ### ⚙️ Configuration
    
    All settings are stored in `.env` file:
    
    - `YOLO_MODEL`: Model path (default: yolov8m.pt)
    - `VIDEO_PATH`: Default input video
    - `FPS_LIMIT`: Processing speed (lower = faster)
    - `DISAPPEAR_THRESHOLD`: Seconds before person is marked as left
    - `WORKER_ZONES`: JSON array of zone polygons
    - `GOOGLE_API_KEY`: For AI agent features
    
    ---
    
    ### 💡 Tips
    
    - Use lower FPS limits for faster processing
    - Define worker zones to improve accuracy
    - Check confusion events to validate tracking quality
    - Use analytics dashboard to understand patterns
    
    ---
    
    ### 🛠️ Troubleshooting
    
    **No videos showing?**
    - Ensure videos are in `data/input/` directory
    - Supported formats: MP4, AVI, MOV, MKV, FLV
    
    **Processing slow?**
    - Reduce FPS limit slider
    - Use lower resolution videos
    - Check if GPU is being used (should see ✅ indicator)
    
    **No confusion events detected?**
    - This is good! No tracking errors found
    - Adjust DISAPPEAR_THRESHOLD if needed
    
    **AI analysis failing?**
    - Ensure `GOOGLE_API_KEY` is set in `.env`
    - Check internet connection
    - Verify video segments are being extracted
    
    """)
    
    st.markdown("---")
    
    st.markdown("### 📊 System Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Configuration:**")
        st.write(f"- YOLO Model: {Config.YOLO_MODEL}")
        st.write(f"- FPS Limit: {Config.FPS_LIMIT}")
        st.write(f"- Disappear Threshold: {Config.DISAPPEAR_THRESHOLD}s")
    
    with col2:
        st.write("**Data Directories:**")
        st.write(f"- Input Videos: {len(get_video_files())} files")
        st.write(f"- Results: {len(get_output_files())} files")
        st.write(f"- Segments: {len(list(Path('data/output/segments').glob('*'))) if Path('data/output/segments').exists() else 0} files")


# ==================== FOOTER ====================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888; font-size: 0.9em;'>
    🏪 ANAVID Queue Intelligence System | Built with Streamlit + YOLO + Google ADK | v1.0
</div>
""", unsafe_allow_html=True)
