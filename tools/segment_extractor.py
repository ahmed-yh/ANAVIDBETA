"""
Extract confusion segments with full context for AI agent
"""

import cv2
import os
import json
from typing import Dict, Any


def extract_segment_with_context(
    video_path: str,
    confusion_event: Dict[str, Any],
    person_tracks: Dict[int, Dict],
    padding_seconds: float = 5.0,
    output_dir: str = "data/output/segments"
) -> Dict[str, Any]:
    """
    Extract video segment around confusion with full tracking context
    
    Args:
        video_path: Path to original video
        confusion_event: Confusion event data
        person_tracks: Dictionary of all person tracking data
        padding_seconds: Seconds before/after confusion to include
        output_dir: Where to save segment
    
    Returns:
        Dictionary with segment path and context for agent
    """
    
    print(f"\n📦 Extracting segment for {confusion_event['event_type']}...")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Calculate frame range
    center_frame = confusion_event['frame_number']
    padding_frames = int(padding_seconds * fps)
    start_frame = max(0, center_frame - padding_frames)
    end_frame = center_frame + padding_frames
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Output filename
    event_type = confusion_event['event_type']
    person_id = confusion_event['person_id']
    timestamp = confusion_event['timestamp']
    
    segment_filename = f"confusion_{event_type}_id{person_id}_t{timestamp:.1f}s.mp4"
    segment_path = os.path.join(output_dir, segment_filename)
    
    # Extract frames
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    writer = cv2.VideoWriter(
        segment_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height)
    )
    
    frame_num = start_frame
    while frame_num <= end_frame and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Add context overlay
        overlay_text = [
            f"CONFUSION: {event_type.upper()}",
            f"Frame: {frame_num} | Time: {frame_num/fps:.1f}s",
        ]
        
        y_pos = 30
        for text in overlay_text:
            cv2.putText(frame, text, (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            y_pos += 30
        
        writer.write(frame)
        frame_num += 1
    
    cap.release()
    writer.release()
    
    # Build context for agent
    # Get IDs involved in confusion
    if event_type == 'id_switch':
        ids_involved = [
            confusion_event['context']['old_id'],
            confusion_event['context']['new_id']
        ]
    else:
        ids_involved = [person_id]
    
    # Get time data for each ID
    id_time_data = {}
    for pid in ids_involved:
        if pid in person_tracks:
            id_time_data[pid] = {
                'time_before_confusion': confusion_event['timestamp'] - person_tracks[pid]['first_seen'],
                'first_seen': person_tracks[pid]['first_seen'],
                'last_seen': person_tracks[pid]['last_seen']
            }
    
    context = {
        'segment_path': segment_path,
        'confusion_type': event_type,
        'frame_range': {
            'start': start_frame,
            'center': center_frame,
            'end': end_frame
        },
        'time_range': {
            'start': start_frame / fps,
            'confusion_at': confusion_event['timestamp'],
            'end': end_frame / fps
        },
        'ids_involved': ids_involved,
        'id_time_data': id_time_data,
        'confusion_details': confusion_event['context']
    }
    
    print(f"✅ Segment extracted: {segment_filename}")
    print(f"   IDs involved: {ids_involved}")
    print(f"   Duration: {(end_frame - start_frame) / fps:.1f}s")
    
    return context
