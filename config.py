"""
Configuration manager - Loads settings from .env file
"""

import os
import json
from typing import List, Tuple
from dotenv import load_dotenv

# Load .env file
load_dotenv()


class Config:
    """Configuration manager for the queue intelligence system"""
    
    # Google API Key
    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY', '')
    
    # Video paths
    VIDEO_PATH = os.getenv('VIDEO_PATH', 'data/input/store_video.mp4')
    OUTPUT_VIDEO_PATH = os.getenv('OUTPUT_VIDEO_PATH', 'data/output/tracked_dwell.mp4')
    
    # YOLO configuration
    YOLO_MODEL = os.getenv('YOLO_MODEL', 'yolov8m.pt')
    DISAPPEAR_THRESHOLD = float(os.getenv('DISAPPEAR_THRESHOLD', '10.0'))
    
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
            print("⚠️ Error parsing WORKER_ZONES from .env, using empty list")
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
        
        # Write back to .env
        with open(env_path, 'w') as f:
            f.writelines(new_lines)
        
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
