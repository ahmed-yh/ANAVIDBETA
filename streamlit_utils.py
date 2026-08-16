"""
Enhanced streaming utilities for real-time YOLO processing in Streamlit
Handles frame extraction, overlay rendering, and live updates
"""

import cv2
import numpy as np
import threading
from typing import Optional, Callable, Tuple
from pathlib import Path
from dataclasses import dataclass
import time


@dataclass
class FrameMetadata:
    """Metadata for each frame during processing"""
    frame_number: int
    timestamp: float
    tracked_ids: list
    confusion_detected: bool
    fps: float


class StreamingVideoProcessor:
    """Real-time video processing with frame updates"""
    
    def __init__(self, video_path: str):
        """Initialize processor"""
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration = self.frame_count / self.fps if self.fps > 0 else 0
        
        self.current_frame_idx = 0
        self.is_playing = False
        
    def get_frame_at(self, frame_number: int) -> Optional[np.ndarray]:
        """Get specific frame"""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        if ret:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return None
    
    def get_next_frame(self) -> Optional[Tuple[np.ndarray, int]]:
        """Get next frame and its index"""
        ret, frame = self.cap.read()
        if ret:
            self.current_frame_idx += 1
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), self.current_frame_idx
        return None, None
    
    def reset(self):
        """Reset to beginning"""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.current_frame_idx = 0
    
    def close(self):
        """Release resources"""
        self.cap.release()
    
    def __del__(self):
        self.close()


def draw_tracked_person(frame: np.ndarray, box: Tuple, person_id: int, 
                       color: Tuple = (0, 255, 0), thickness: int = 2) -> np.ndarray:
    """Draw bounding box with person ID"""
    x1, y1, x2, y2 = box
    
    # Draw box
    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
    
    # Draw ID label
    label = f"ID: {person_id}"
    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
    
    cv2.rectangle(
        frame,
        (int(x1), int(y1) - label_size[1] - 4),
        (int(x1) + label_size[0] + 4, int(y1)),
        color,
        -1
    )
    
    cv2.putText(
        frame,
        label,
        (int(x1) + 2, int(y1) - 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )
    
    return frame


def draw_confusion_indicator(frame: np.ndarray, confusion_type: str, 
                            person_id: int) -> np.ndarray:
    """Draw confusion warning overlay"""
    height, width = frame.shape[:2]
    
    # Semi-transparent red overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, 100), (0, 0, 255), -1)
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
    
    # Warning text
    warning_text = f"⚠️ CONFUSION DETECTED: {confusion_type.upper()} (ID: {person_id})"
    cv2.putText(
        frame,
        warning_text,
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        3
    )
    
    return frame


def draw_zone_polygon(frame: np.ndarray, zone: list, color: Tuple = (255, 0, 0),
                     alpha: float = 0.3, thickness: int = 2) -> np.ndarray:
    """Draw exclusion zone polygon"""
    if len(zone) < 2:
        return frame
    
    points = np.array(zone, dtype=np.int32)
    
    # Draw filled polygon
    overlay = frame.copy()
    cv2.fillPoly(overlay, [points], color)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    
    # Draw outline
    cv2.polylines(frame, [points], True, color, thickness)
    
    return frame


def add_stats_overlay(frame: np.ndarray, stats: dict, position: str = "top-left") -> np.ndarray:
    """Add statistics text overlay"""
    height, width = frame.shape[:2]
    
    # Create semi-transparent background
    overlay = frame.copy()
    
    y_start = 30 if position == "top-left" else height - 150
    
    cv2.rectangle(overlay, (10, y_start - 20), (350, y_start + 120), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    # Draw text
    y_pos = y_start
    for key, value in stats.items():
        text = f"{key}: {value}"
        cv2.putText(
            frame,
            text,
            (20, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            1
        )
        y_pos += 30
    
    return frame


def create_comparison_frame(frame1: np.ndarray, frame2: np.ndarray,
                           label1: str = "Original", label2: str = "Processed") -> np.ndarray:
    """Create side-by-side comparison"""
    h1, w1 = frame1.shape[:2]
    h2, w2 = frame2.shape[:2]
    
    # Resize to match height
    target_height = max(h1, h2)
    
    if h1 != target_height:
        frame1 = cv2.resize(frame1, (int(w1 * target_height / h1), target_height))
    if h2 != target_height:
        frame2 = cv2.resize(frame2, (int(w2 * target_height / h2), target_height))
    
    # Create combined frame
    combined = np.hstack([frame1, frame2])
    
    # Add labels
    cv2.putText(
        combined,
        label1,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )
    
    cv2.putText(
        combined,
        label2,
        (frame1.shape[1] + 20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )
    
    return combined


class LogBuffer:
    """Thread-safe log buffer for real-time updates"""
    
    def __init__(self, max_lines: int = 100):
        self.logs = []
        self.max_lines = max_lines
        self.lock = threading.Lock()
    
    def add(self, message: str, level: str = "INFO"):
        """Add log message"""
        with self.lock:
            timestamp = time.strftime("%H:%M:%S")
            log_entry = f"[{timestamp}] [{level}] {message}"
            self.logs.append(log_entry)
            
            # Keep only recent logs
            if len(self.logs) > self.max_lines:
                self.logs = self.logs[-self.max_lines:]
    
    def get_all(self) -> list:
        """Get all logs"""
        with self.lock:
            return self.logs.copy()
    
    def clear(self):
        """Clear all logs"""
        with self.lock:
            self.logs = []
    
    def get_formatted(self) -> str:
        """Get formatted log string"""
        return "\n".join(self.get_all())


def create_heatmap_frame(frame: np.ndarray, positions: list, decay: float = 0.95) -> np.ndarray:
    """Create heatmap overlay from position history"""
    height, width = frame.shape[:2]
    
    # Create heatmap
    heatmap = np.zeros((height, width), dtype=np.float32)
    
    # Add gaussian blobs at each position
    for pos in positions:
        x, y = int(pos[0]), int(pos[1])
        if 0 <= x < width and 0 <= y < height:
            # Draw gaussian
            cv2.circle(heatmap, (x, y), 30, 1.0, -1)
    
    # Apply colormap
    heatmap = (heatmap * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Blend with original
    result = cv2.addWeighted(frame, 0.7, heatmap_colored, 0.3, 0)
    
    return result


def draw_trajectory(frame: np.ndarray, trajectory: list, person_id: int,
                   color: Tuple = (0, 255, 255), thickness: int = 2) -> np.ndarray:
    """Draw person's trajectory"""
    if len(trajectory) < 2:
        return frame
    
    points = np.array(trajectory, dtype=np.int32)
    
    # Draw lines between points
    for i in range(len(points) - 1):
        cv2.line(frame, tuple(points[i]), tuple(points[i + 1]), color, thickness)
    
    # Draw circle at start
    cv2.circle(frame, tuple(points[0]), 5, (0, 255, 0), -1)
    
    # Draw circle at end
    cv2.circle(frame, tuple(points[-1]), 5, (0, 0, 255), -1)
    
    # Add label
    cv2.putText(
        frame,
        f"ID {person_id}",
        tuple(points[-1] + 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2
    )
    
    return frame
