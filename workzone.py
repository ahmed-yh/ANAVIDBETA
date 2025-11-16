"""
Interactive tool to define worker/staff exclusion zones
Auto-saves to .env file
"""

import cv2
import numpy as np
import sys
from config import Config


class ZoneDefiner:
    def __init__(self, video_path, max_display_width=1280, max_display_height=720):
        self.video_path = video_path
        self.points = []
        self.zones = []
        self.current_zone_points = []
        
        # Get video properties
        cap = cv2.VideoCapture(video_path)
        self.original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        # Calculate display scale
        width_scale = max_display_width / self.original_width
        height_scale = max_display_height / self.original_height
        self.scale = min(width_scale, height_scale, 1.0)
        
        self.display_width = int(self.original_width * self.scale)
        self.display_height = int(self.original_height * self.scale)
        
        print(f"\n=== VIDEO INFO ===")
        print(f"Original: {self.original_width}x{self.original_height}")
        print(f"Display: {self.display_width}x{self.display_height}")
        print(f"Scale: {self.scale:.2f}")
        print(f"==================\n")
        
    def scale_point_to_original(self, x, y):
        """Convert display coordinates to original video coordinates"""
        return (int(x / self.scale), int(y / self.scale))
    
    def scale_point_to_display(self, x, y):
        """Convert original coordinates to display coordinates"""
        return (int(x * self.scale), int(y * self.scale))
        
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            orig_x, orig_y = self.scale_point_to_original(x, y)
            self.current_zone_points.append((orig_x, orig_y))
            print(f"Point added: Display({x}, {y}) -> Original({orig_x}, {orig_y})")
            
        elif event == cv2.EVENT_RBUTTONDOWN:
            if len(self.current_zone_points) >= 3:
                self.zones.append(self.current_zone_points.copy())
                print(f"✓ Zone {len(self.zones)} completed with {len(self.current_zone_points)} points")
                self.current_zone_points = []
            else:
                print(f"✗ Need at least 3 points (have {len(self.current_zone_points)})")
    
    def define_zones(self):
        cap = cv2.VideoCapture(self.video_path)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            print("❌ Error reading video")
            return None
        
        frame_display = cv2.resize(frame, (self.display_width, self.display_height))
        clone = frame_display.copy()
        
        cv2.namedWindow("Define Worker Zones", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Define Worker Zones", self.display_width, self.display_height)
        cv2.setMouseCallback("Define Worker Zones", self.mouse_callback)
        
        print("\n" + "="*60)
        print("WORKER ZONE DEFINITION TOOL")
        print("="*60)
        print("LEFT CLICK:  Add point to current zone")
        print("RIGHT CLICK: Complete current zone")
        print("Press 'r':   Reset all zones")
        print("Press 'u':   Undo last point")
        print("Press 'q':   Finish and save to .env")
        print("="*60 + "\n")
        
        while True:
            display = frame_display.copy()
            
            # Draw completed zones
            for i, zone in enumerate(self.zones):
                zone_display = [self.scale_point_to_display(x, y) for x, y in zone]
                zone_array = np.array(zone_display, dtype=np.int32)
                
                overlay = display.copy()
                cv2.fillPoly(overlay, [zone_array], (0, 0, 255))
                display = cv2.addWeighted(display, 0.7, overlay, 0.3, 0)
                
                cv2.polylines(display, [zone_array], True, (0, 0, 255), 3)
                
                centroid = np.mean(zone_array, axis=0).astype(int)
                cv2.putText(display, f"WORKER ZONE {i+1}", tuple(centroid),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
                cv2.putText(display, f"WORKER ZONE {i+1}", tuple(centroid),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Draw current zone being defined
            if len(self.current_zone_points) > 0:
                display_points = [self.scale_point_to_display(x, y) for x, y in self.current_zone_points]
                
                for point in display_points:
                    cv2.circle(display, point, 7, (0, 255, 0), -1)
                    cv2.circle(display, point, 7, (255, 255, 255), 2)
                
                if len(display_points) > 1:
                    zone_array = np.array(display_points, dtype=np.int32)
                    cv2.polylines(display, [zone_array], False, (0, 255, 0), 3)
                
                cv2.putText(display, f"Points: {len(self.current_zone_points)}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Instructions overlay
            instructions = [
                f"Zones: {len(self.zones)}",
                f"Current: {len(self.current_zone_points)} pts",
            ]
            y_offset = self.display_height - 60
            for text in instructions:
                cv2.putText(display, text, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                y_offset += 30
            
            cv2.imshow("Define Worker Zones", display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('r'):
                frame_display = clone.copy()
                self.current_zone_points = []
                self.zones = []
                print("🔄 Reset all zones")
                
            elif key == ord('u'):
                if len(self.current_zone_points) > 0:
                    removed = self.current_zone_points.pop()
                    print(f"⬅️  Undo: Removed point {removed}")
                else:
                    print("⚠️  Nothing to undo")
                    
            elif key == ord('q'):
                break
        
        cv2.destroyAllWindows()
        
        if self.zones:
            print("\n" + "="*60)
            print("✅ ZONES DEFINED SUCCESSFULLY")
            print("="*60)
            
            # SAVE TO .ENV FILE
            Config.update_worker_zones(self.zones)
            
            print("\n📋 Zones saved to .env file!")
            print(f"   {len(self.zones)} zone(s) defined")
            
            for i, zone in enumerate(self.zones, 1):
                print(f"   Zone {i}: {len(zone)} points")
            
            print("\n💡 You can now use these zones in your tracker automatically!")
            print("="*60 + "\n")
            
            return self.zones
        else:
            print("\n⚠️  No zones defined")
            return None


if __name__ == "__main__":
    if len(sys.argv) != 2:
        # Try to use video path from config
        video_path = Config.VIDEO_PATH
        if not video_path or not os.path.exists(video_path):
            print("Usage: python workzone.py <video_path>")
            print("Or set VIDEO_PATH in .env file")
            sys.exit(1)
    else:
        video_path = sys.argv[1]
    
    import os
    if not os.path.exists(video_path):
        print(f"❌ Video not found: {video_path}")
        sys.exit(1)
    
    definer = ZoneDefiner(video_path)
    definer.define_zones()
