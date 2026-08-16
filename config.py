"""
Configuration manager - Loads settings from .env file
"""

import os
import json
import logging
import tempfile
from typing import List, Tuple
from dotenv import load_dotenv

# Load .env file
load_dotenv()

logger = logging.getLogger(__name__)


class Config:
    """Configuration manager for the queue intelligence system"""

    # Google API Key
    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY', '')

    # Video paths
    VIDEO_PATH = os.getenv('VIDEO_PATH', 'data/input/result3.mp4')
    OUTPUT_VIDEO_PATH = os.getenv('OUTPUT_VIDEO_PATH', 'data/output/tracked_dwell.mp4')

    # YOLO configuration
    YOLO_MODEL = os.getenv('YOLO_MODEL', 'yolov8m.pt')
    DISAPPEAR_THRESHOLD = float(os.getenv('DISAPPEAR_THRESHOLD', '10.0'))
    FPS_LIMIT = int(os.getenv('FPS_LIMIT', '15'))
    YOLO_CONF = float(os.getenv('YOLO_CONF', '0.4'))
    YOLO_IOU = float(os.getenv('YOLO_IOU', '0.5'))
    # These are deliberately permissive: every candidate they flag gets a clip
    # cut and sent to Gemini for a real judgment call - Python's job here is
    # just to notice "something happened", not to pre-filter what's worth AI
    # review. Raise them only if API cost/volume becomes a real problem.
    CONFUSION_MATCH_DISTANCE_PX = float(os.getenv('CONFUSION_MATCH_DISTANCE_PX', '80'))
    CONFUSION_DEBOUNCE_FRAMES = int(os.getenv('CONFUSION_DEBOUNCE_FRAMES', '2'))
    OCCLUSION_MIN_HIDDEN_SECONDS = float(os.getenv('OCCLUSION_MIN_HIDDEN_SECONDS', '0.0'))

    # Performance: downscale frames before inference/drawing/writing (YOLO
    # internally resizes to its own imgsz anyway, so this mostly saves on
    # frame-read/zone-check/draw/encode overhead, not detection quality).
    # Set to 0 to disable and process at native resolution.
    PROCESS_WIDTH = int(os.getenv('PROCESS_WIDTH', '1280'))

    # Tracker: raised track_buffer so brief occlusions in busy scenes don't
    # immediately spawn a new ID. See trackers/custom_botsort.yaml.
    TRACKER_CONFIG = os.getenv('TRACKER_CONFIG', 'trackers/custom_botsort.yaml')

    # Worker zones
    @staticmethod
    def get_worker_zones() -> List[List[Tuple[int, int]]]:
        """
        Load worker zones from .env file
        Returns list of polygons
        """
        zones_json = os.getenv('WORKER_ZONES', '[]')
        try:
            zones_raw = json.loads(zones_json)
            # Convert to list of list of tuples
            zones = [
                [tuple(point) for point in zone]
                for zone in zones_raw
            ]
            return zones
        except json.JSONDecodeError:
            logger.warning("Could not parse WORKER_ZONES from .env (invalid JSON); using empty list")
            return []
    
    @staticmethod
    def update_worker_zones(zones: List[List[Tuple[int, int]]]):
        """
        Update worker zones in .env file
        
        Args:
            zones: List of polygons (list of list of tuples)
        """
        # Convert tuples to lists for JSON serialization
        zones_serializable = [
            [[int(x), int(y)] for x, y in zone]
            for zone in zones
        ]
        
        zones_json = json.dumps(zones_serializable)
        
        # Read current .env file
        env_path = '.env'
        
        if os.path.exists(env_path):
            with open(env_path, 'r') as f:
                lines = f.readlines()
        else:
            lines = []
        
        # Update or add WORKER_ZONES line
        updated = False
        new_lines = []
        
        for line in lines:
            if line.startswith('WORKER_ZONES='):
                new_lines.append(f'WORKER_ZONES={zones_json}\n')
                updated = True
            else:
                new_lines.append(line)
        
        # If WORKER_ZONES wasn't in file, add it
        if not updated:
            new_lines.append(f'\nWORKER_ZONES={zones_json}\n')

        # Write atomically: write to a temp file in the same directory, then
        # replace .env in one step, so a crash mid-write can't corrupt it.
        env_dir = os.path.dirname(os.path.abspath(env_path)) or '.'
        fd, tmp_path = tempfile.mkstemp(dir=env_dir, prefix='.env.', suffix='.tmp')
        try:
            with os.fdopen(fd, 'w') as f:
                f.writelines(new_lines)
            os.replace(tmp_path, env_path)
        except Exception:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise

        # load_dotenv() only populates os.environ once at import time and
        # never overwrites existing keys, so without this, a long-running
        # process (e.g. Streamlit) would keep reading the OLD zone from
        # memory even though the file on disk is already correct.
        os.environ['WORKER_ZONES'] = zones_json

        print(f"✅ Worker zones updated in .env file")
    
    @staticmethod
    def validate():
        """Validate configuration"""
        errors = []
        
        if not Config.GOOGLE_API_KEY:
            errors.append("GOOGLE_API_KEY not set in .env file")
        
        if not os.path.exists(Config.VIDEO_PATH):
            errors.append(f"Video file not found: {Config.VIDEO_PATH}")
        
        if errors:
            print("\n⚠️ Configuration Errors:")
            for error in errors:
                print(f"   - {error}")
            return False
        
        return True
    
    @staticmethod
    def print_config():
        """Print current configuration"""
        print("\n" + "="*60)
        print("📋 CURRENT CONFIGURATION")
        print("="*60)
        print(f"API Key: {'✅ Set' if Config.GOOGLE_API_KEY else '❌ Not set'}")
        print(f"Video Path: {Config.VIDEO_PATH}")
        print(f"Output Path: {Config.OUTPUT_VIDEO_PATH}")
        print(f"YOLO Model: {Config.YOLO_MODEL}")
        print(f"Disappear Threshold: {Config.DISAPPEAR_THRESHOLD}s")
        
        zones = Config.get_worker_zones()
        print(f"Worker Zones: {len(zones)} zone(s) defined")
        
        for i, zone in enumerate(zones, 1):
            print(f"  Zone {i}: {len(zone)} points")
        
        print("="*60 + "\n")


if __name__ == "__main__":
    # Test configuration
    Config.print_config()
    
    if Config.validate():
        print("✅ Configuration is valid!")
    else:
        print("❌ Configuration has errors!")
