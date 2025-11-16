"""
Enhanced Dwell Time Tracker with Confusion Detection
Tracks people and detects when YOLO gets confused
Now uses .env configuration file
"""

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from collections import defaultdict
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional
import time
import json
import os

# Import configuration manager
from config import Config


@dataclass
class ConfusionEvent:
    """Tracks when YOLO gets confused"""
    event_type: str  # 'id_switch', 'occlusion', 'return_after_leave'
    person_id: int
    frame_number: int
    timestamp: float
    context: dict
    segment_start_frame: int = 0
    segment_end_frame: int = 0
    
    def to_dict(self):
        return asdict(self)


class DwellTimeTracker:
    def __init__(self, model_path='yolov8m.pt', disappear_threshold=10.0, exclude_zones=None):
        """
        Initialize dwell time tracker with confusion detection
        
        Args:
            model_path: Path to YOLO model
            disappear_threshold: Seconds before considering person as exited
            exclude_zones: List of polygons defining worker/staff areas
        """
        print(f"Loading YOLO model: {model_path}")
        self.model = YOLO(model_path)
        
        # GPU detection
        import torch
        self.device = 0 if torch.cuda.is_available() else 'cpu'
        if torch.cuda.is_available():
            print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            print(f"⚠️ Using CPU (slower)")
        
        self.disappear_threshold = disappear_threshold
        self.exclude_zones = exclude_zones or []
        
        if self.exclude_zones:
            print(f"✅ Worker zones configured: {len(self.exclude_zones)} zone(s)")
        else:
            print(f"⚠️ No worker zones configured - all people will be tracked")
        
        # Tracking data structures
        self.person_first_seen = {}
        self.person_last_seen = {}
        self.person_total_time = {}
        self.person_disappeared = {}
        
        # Complete tracking data (for agent analysis)
        self.person_tracks = {}  # person_id -> {first_seen, last_seen, total_time}
        
        # Worker tracking
        self.worker_ids = set()
        
        # Results storage
        self.results_data = []
        self.active_ids = set()
        
        # CONFUSION DETECTION
        self.confusion_events = []
        self.previous_frame_data = {}
        self.id_switch_candidates = {}
        
    def is_in_exclude_zone(self, box):
        """Check if bounding box center is in any exclude zone"""
        if not self.exclude_zones:
            return False
        
        x1, y1, x2, y2 = box
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        
        for zone in self.exclude_zones:
            zone_array = np.array(zone, dtype=np.int32)
            if cv2.pointPolygonTest(zone_array, (center_x, center_y), False) >= 0:
                return True
        
        return False
    
    def detect_confusion(self, current_ids, track_ids, boxes, confidences, current_time, frame_count):
        """
        Detect confusion events while processing video
        Returns: List of ConfusionEvent objects
        """
        confusion_events = []
        
        # 1. DETECT ID SWITCHES
        if hasattr(self, 'previous_frame_data') and self.previous_frame_data:
            prev_ids = set(self.previous_frame_data.keys())
            curr_ids = set(current_ids)
            
            disappeared = prev_ids - curr_ids
            appeared = curr_ids - prev_ids
            
            # If same number disappeared and appeared = potential ID switch
            if len(disappeared) == 1 and len(appeared) == 1:
                old_id = list(disappeared)[0]
                new_id = list(appeared)[0]
                
                # Check if new ID is near old ID's last position
                old_pos = self.previous_frame_data[old_id]['bbox']
                
                # Find new ID's position
                new_pos = None
                for i, tid in enumerate(track_ids):
                    if tid == new_id:
                        new_pos = boxes[i]
                        break
                
                if new_pos is not None:
                    # Calculate distance between centers
                    old_center = ((old_pos[0]+old_pos[2])/2, (old_pos[1]+old_pos[3])/2)
                    new_center = ((new_pos[0]+new_pos[2])/2, (new_pos[1]+new_pos[3])/2)
                    distance = ((old_center[0]-new_center[0])**2 + (old_center[1]-new_center[1])**2)**0.5
                    
                    if distance < 150:  # pixels - they're close!
                        confusion_events.append(ConfusionEvent(
                            event_type='id_switch',
                            person_id=new_id,
                            frame_number=frame_count,
                            timestamp=current_time,
                            context={
                                'old_id': int(old_id), 
                                'new_id': int(new_id), 
                                'distance': float(distance),
                                'old_bbox': old_pos.tolist(),
                                'new_bbox': new_pos.tolist()
                            }
                        ))
                        print(f"  ⚠️ ID SWITCH: {old_id} → {new_id} (distance: {distance:.0f}px)")
        
        # 2. DETECT OCCLUSIONS (person disappeared briefly)
        for person_id in list(self.person_disappeared.keys()):
            if person_id in current_ids:
                # They're back!
                time_gone = current_time - self.person_disappeared[person_id]
                if time_gone < 3.0:  # Short disappearance = occlusion
                    confusion_events.append(ConfusionEvent(
                        event_type='occlusion',
                        person_id=int(person_id),
                        frame_number=frame_count,
                        timestamp=current_time,
                        context={'time_hidden': float(time_gone)}
                    ))
                    print(f"  ⚠️ OCCLUSION: Person {person_id} hidden for {time_gone:.1f}s")
        
        # 3. DETECT RETURNS AFTER LEAVING
        for person_id in current_ids:
            if person_id in self.person_disappeared:
                time_gone = current_time - self.person_disappeared[person_id]
                if time_gone > self.disappear_threshold:
                    confusion_events.append(ConfusionEvent(
                        event_type='return_after_leave',
                        person_id=int(person_id),
                        frame_number=frame_count,
                        timestamp=current_time,
                        context={'time_away': float(time_gone)}
                    ))
                    print(f"  ⚠️ RETURN: Person {person_id} came back after {time_gone:.1f}s")
        
        # Store current frame data for next comparison
        self.previous_frame_data = {}
        for i, tid in enumerate(track_ids):
            if tid in current_ids:
                self.previous_frame_data[tid] = {'bbox': boxes[i]}
        
        return confusion_events
        
    def process_video(self, video_path, output_path=None, fps_limit=30, show_preview=True):
        """
        Process video and track dwell time for all people
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"\n{'='*60}")
        print(f"Video Properties:")
        print(f"  Resolution: {width}x{height}")
        print(f"  FPS: {fps}")
        print(f"  Total Frames: {total_frames}")
        print(f"  Duration: {total_frames/fps:.1f} seconds")
        print(f"  Disappear Threshold: {self.disappear_threshold}s")
        print(f"{'='*60}\n")
        
        # Setup video writer if output requested
        writer = None
        if output_path:
            # Create output directory if needed
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        start_time = time.time()
        
        # Process frame by frame
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Skip frames to match fps_limit
            if frame_count % max(1, fps // fps_limit) != 0:
                continue
            
            current_time = frame_count / fps
            
            # Run YOLO detection with tracking
            results = self.model.track(
                frame, 
                persist=True,
                classes=[0],
                conf=0.4,
                iou=0.5,
                verbose=False,
                device=self.device
            )
            
            visible_ids_this_frame = set()
            
            # Process detections
            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                confidences = results[0].boxes.conf.cpu().numpy()
                
                # Collect current IDs (excluding workers)
                current_ids_list = []
                for box, track_id, conf in zip(boxes, track_ids, confidences):
                    is_worker = self.is_in_exclude_zone(box)
                    if not is_worker:
                        visible_ids_this_frame.add(track_id)
                        current_ids_list.append(track_id)
                
                # DETECT CONFUSION EVENTS
                new_confusions = self.detect_confusion(
                    current_ids_list, track_ids, boxes, confidences, current_time, frame_count
                )
                self.confusion_events.extend(new_confusions)
                
                # Normal tracking processing
                for box, track_id, conf in zip(boxes, track_ids, confidences):
                    
                    is_worker = self.is_in_exclude_zone(box)
                    
                    if is_worker:
                        self.worker_ids.add(track_id)
                        
                        if writer or show_preview:
                            x1, y1, x2, y2 = box.astype(int)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (128, 128, 128), 2)
                            cv2.putText(frame, f"STAFF {track_id}", (x1, y1-10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)
                        continue
                    
                    visible_ids_this_frame.add(track_id)
                    
                    # First time seeing this person
                    if track_id not in self.person_first_seen:
                        self.person_first_seen[track_id] = current_time
                        self.person_last_seen[track_id] = current_time
                        self.person_total_time[track_id] = 0
                        self.active_ids.add(track_id)
                        
                        # ADD TO PERSON_TRACKS (for agent)
                        self.person_tracks[track_id] = {
                            'first_seen': current_time,
                            'last_seen': current_time,
                            'total_time': 0
                        }
                        
                        print(f"[{current_time:.1f}s] Customer {track_id} entered")
                    
                    # Update last seen
                    self.person_last_seen[track_id] = current_time
                    
                    # UPDATE PERSON_TRACKS (for agent)
                    self.person_tracks[track_id]['last_seen'] = current_time
                    self.person_tracks[track_id]['total_time'] = current_time - self.person_tracks[track_id]['first_seen']
                    
                    if track_id in self.person_disappeared:
                        disappear_duration = current_time - self.person_disappeared[track_id]
                        if disappear_duration < self.disappear_threshold:
                            print(f"[{current_time:.1f}s] Customer {track_id} reappeared after {disappear_duration:.1f}s")
                        del self.person_disappeared[track_id]
                    
                    # Draw on frame
                    if writer or show_preview:
                        x1, y1, x2, y2 = box.astype(int)
                        current_dwell = current_time - self.person_first_seen[track_id]
                        
                        if current_dwell < 60:
                            color = (0, 255, 0)
                        elif current_dwell < 180:
                            color = (0, 255, 255)
                        else:
                            color = (0, 0, 255)
                        
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        label = f"ID:{track_id} - {current_dwell:.0f}s"
                        cv2.putText(frame, label, (x1, y1-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Check for people who disappeared
            disappeared_ids = self.active_ids - visible_ids_this_frame
            for track_id in disappeared_ids:
                if track_id not in self.person_disappeared:
                    self.person_disappeared[track_id] = current_time
                else:
                    disappear_duration = current_time - self.person_disappeared[track_id]
                    if disappear_duration >= self.disappear_threshold:
                        total_time = self.person_last_seen[track_id] - self.person_first_seen[track_id]
                        self.person_total_time[track_id] = total_time
                        self.active_ids.remove(track_id)
                        
                        self.results_data.append({
                            'person_id': track_id,
                            'first_seen': self.person_first_seen[track_id],
                            'last_seen': self.person_last_seen[track_id],
                            'total_time': total_time,
                            'status': 'exited'
                        })
                        
                        print(f"[{current_time:.1f}s] Customer {track_id} EXITED - Total time: {total_time:.1f}s")
                        del self.person_disappeared[track_id]
            
            self.active_ids = visible_ids_this_frame.copy()
            
            # Add statistics overlay
            if writer or show_preview:
                # Draw worker zones
                if self.exclude_zones:
                    overlay = frame.copy()
                    for zone in self.exclude_zones:
                        zone_array = np.array(zone, dtype=np.int32)
                        cv2.polylines(frame, [zone_array], True, (0, 0, 255), 2)
                        cv2.fillPoly(overlay, [zone_array], (0, 0, 255))
                    frame = cv2.addWeighted(frame, 0.9, overlay, 0.1, 0)
                
                # Draw stats overlay
                overlay = frame.copy()
                cv2.rectangle(overlay, (10, 10), (450, 210), (0, 0, 0), -1)
                frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
                
                stats_text = [
                    f"Time: {current_time:.1f}s",
                    f"Customers Visible: {len(visible_ids_this_frame)}",
                    f"Workers: {len(self.worker_ids)}",
                    f"Total Tracked: {len(self.person_first_seen)}",
                    f"Completed: {len(self.results_data)}",
                    f"Confusions: {len(self.confusion_events)}",
                ]
                
                y_offset = 30
                for text in stats_text:
                    cv2.putText(frame, text, (20, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    y_offset += 30
                
                if writer:
                    writer.write(frame)
                
                if show_preview:
                    cv2.namedWindow("Dwell Time Tracking", cv2.WINDOW_NORMAL)
                    cv2.resizeWindow("Dwell Time Tracking", 1280, 720)
                    cv2.imshow("Dwell Time Tracking", frame)
                    if cv2.waitKey(33) & 0xFF == ord('q'):
                        break
            
            # Progress update
            if frame_count % 100 == 0:
                elapsed = time.time() - start_time
                fps_processing = frame_count / elapsed
                progress = (frame_count / total_frames) * 100
                print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames}) - {fps_processing:.1f} fps")
        
        # Process people still visible at end
        final_time = total_frames / fps
        for track_id in self.active_ids:
            if track_id in self.person_first_seen:
                total_time = final_time - self.person_first_seen[track_id]
                self.person_total_time[track_id] = total_time
                
                # UPDATE PERSON_TRACKS for people still visible
                self.person_tracks[track_id]['last_seen'] = final_time
                self.person_tracks[track_id]['total_time'] = total_time
                
                self.results_data.append({
                    'person_id': track_id,
                    'first_seen': self.person_first_seen[track_id],
                    'last_seen': final_time,
                    'total_time': total_time,
                    'status': 'still_visible'
                })
                
                print(f"[END] Customer {track_id} still visible - Total time: {total_time:.1f}s")
        
        # Cleanup
        cap.release()
        if writer:
            writer.release()
        if show_preview:
            cv2.destroyAllWindows()
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"PROCESSING COMPLETE!")
        print(f"{'='*60}")
        print(f"Total customers tracked: {len(self.person_first_seen)}")
        print(f"Workers identified: {len(self.worker_ids)}")
        print(f"Customers who exited: {len([r for r in self.results_data if r['status'] == 'exited'])}")
        print(f"Customers still visible: {len([r for r in self.results_data if r['status'] == 'still_visible'])}")
        
        # CONFUSION REPORT
        print(f"\n{'='*60}")
        print(f"CONFUSION EVENTS DETECTED: {len(self.confusion_events)}")
        print(f"{'='*60}")
        
        confusion_summary = defaultdict(int)
        for event in self.confusion_events:
            confusion_summary[event.event_type] += 1
        
        for event_type, count in confusion_summary.items():
            print(f"  {event_type}: {count}")
        
        print(f"\nDetailed Confusion Log:")
        for event in self.confusion_events:
            print(f"  [{event.timestamp:.1f}s] {event.event_type} - Person {event.person_id}")
        print(f"{'='*60}\n")
        
        if self.results_data:
            avg_time = np.mean([r['total_time'] for r in self.results_data])
            max_time = np.max([r['total_time'] for r in self.results_data])
            min_time = np.min([r['total_time'] for r in self.results_data])
            print(f"\nDwell Time Statistics (Customers Only):")
            print(f"  Average: {avg_time:.1f}s ({avg_time/60:.1f} minutes)")
            print(f"  Maximum: {max_time:.1f}s ({max_time/60:.1f} minutes)")
            print(f"  Minimum: {min_time:.1f}s ({min_time/60:.1f} minutes)")
        print(f"{'='*60}\n")
    
    def save_results(self, output_csv='results/dwell_time_results.csv'):
        """Save tracking results to CSV"""
        if not self.results_data:
            print("No results to save!")
            return None
        
        df = pd.DataFrame(self.results_data)
        df['total_time_minutes'] = df['total_time'] / 60
        df = df.sort_values('total_time', ascending=False)
        
        # Create directory if needed
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        
        df.to_csv(output_csv, index=False)
        print(f"✅ Results saved to {output_csv}")
        return df
    
    def save_confusion_report(self, output_json='results/confusion_report.json'):
        """Save confusion events to JSON"""
        if not self.confusion_events:
            print("No confusion events to save!")
            return
        
        os.makedirs(os.path.dirname(output_json), exist_ok=True)
        
        report = {
            'total_confusions': len(self.confusion_events),
            'events': [event.to_dict() for event in self.confusion_events]
        }
        
        with open(output_json, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"✅ Confusion report saved to {output_json}")


# Main execution
if __name__ == "__main__":
    import sys
    
    # Print current configuration from .env
    print("\n" + "="*60)
    print("🔧 LOADING CONFIGURATION FROM .ENV")
    print("="*60)
    Config.print_config()
    
    # Override video path from command line if provided
    if len(sys.argv) > 1:
        VIDEO_PATH = sys.argv[1]
        print(f"📹 Using command line video: {VIDEO_PATH}")
    else:
        VIDEO_PATH = Config.VIDEO_PATH
        print(f"📹 Using .env video: {VIDEO_PATH}")
    
    # Check if video exists
    if not os.path.exists(VIDEO_PATH):
        print(f"\n❌ Video file not found: {VIDEO_PATH}")
        print("Please:")
        print("  1. Place your video at the path specified in .env")
        print("  2. Or run: python queue_tracker.py <path-to-video>")
        sys.exit(1)
    
    # Load worker zones from .env
    WORKER_ZONES = Config.get_worker_zones()
    
    if not WORKER_ZONES:
        print("\n⚠️ WARNING: No worker zones defined!")
        print("Run 'python workzone.py' to define worker zones")
        print("Continuing without worker zone filtering...\n")
    
    # Create tracker with configuration from .env
    tracker = DwellTimeTracker(
        model_path=Config.YOLO_MODEL,
        disappear_threshold=Config.DISAPPEAR_THRESHOLD,
        exclude_zones=WORKER_ZONES
    )
    
    # Process video
    tracker.process_video(
        VIDEO_PATH, 
        output_path=Config.OUTPUT_VIDEO_PATH,
        fps_limit=5,
        show_preview=True
    )
    
    # Save results
    df = tracker.save_results("results/dwell_time_results.csv")
    tracker.save_confusion_report("results/confusion_report.json")
    
    # Display results
    if df is not None and len(df) > 0:
        print("\n=== TOP 10 LONGEST DWELL TIMES ===")
        print(df[['person_id', 'total_time', 'total_time_minutes', 'status']].head(10).to_string(index=False))
