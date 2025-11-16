"""
Main entry point: Run the complete queue intelligence system
FIXED: NumPy int64 JSON serialization error
"""

import os
import json
from config import Config
from queue_tracker import DwellTimeTracker
from tools.segment_extractor import extract_segment_with_context
from agent import create_segment_analyzer_agent, analyze_confusion_segment


def main():
    """
    Run complete queue intelligence system
    """
    
    print("="*60)
    print("🏪 QUEUE INTELLIGENCE SYSTEM")
    print("="*60)
    
    # Load and validate configuration
    print("\n" + "="*60)
    print("📋 CURRENT CONFIGURATION")
    print("="*60)
    Config.print_config()
    print("="*60)
    
    if not Config.validate():
        print("\n❌ Configuration errors - please fix .env file")
        return
    
    VIDEO_PATH = Config.VIDEO_PATH
    WORKER_ZONES = Config.get_worker_zones()
    
    # STEP 1: Run YOLO tracker with confusion detection
    print("\n📹 STEP 1: Running YOLO tracker...")
    tracker = DwellTimeTracker(
        model_path=Config.YOLO_MODEL,
        disappear_threshold=Config.DISAPPEAR_THRESHOLD,
        exclude_zones=WORKER_ZONES
    )
    
    tracker.process_video(
        VIDEO_PATH,
        output_path=Config.OUTPUT_VIDEO_PATH,
        fps_limit=10,
        show_preview=True
    )
    
    tracker.save_results("results/initial_tracking.csv")
    tracker.save_confusion_report("results/confusion_events.json")
    
    print(f"\n✅ Initial tracking complete")
    print(f"   Customers: {len(tracker.person_first_seen)}")
    print(f"   Confusions detected: {len(tracker.confusion_events)}")
    
    if len(tracker.confusion_events) == 0:
        print("\n🎉 No confusions detected! All times are accurate.")
        return
    
    # STEP 2: Create AI agent
    print(f"\n🤖 STEP 2: Creating AI agent...")
    agent = create_segment_analyzer_agent()
    
    # STEP 3: Process each confusion
    print(f"\n🔍 STEP 3: Analyzing confusion segments...")
    
    corrections = []
    
    for i, confusion in enumerate(tracker.confusion_events, 1):
        print(f"\n{'='*60}")
        print(f"Confusion {i}/{len(tracker.confusion_events)}")
        print(f"{'='*60}")
        
        segment_context = extract_segment_with_context(
            video_path=VIDEO_PATH,
            confusion_event=confusion.to_dict(),
            person_tracks=tracker.person_tracks,
            padding_seconds=5.0
        )
        
        correction = analyze_confusion_segment(segment_context, agent)
        
        corrections.append({
            'confusion_event': confusion.to_dict(),
            'segment_context': segment_context,
            'agent_correction': correction
        })
    
    # STEP 4: Apply corrections and save
    print(f"\n💾 STEP 4: Applying corrections...")
    
    final_times = {}
    
    # ✅ FIX: Convert all NumPy types to Python types
    for person_id, track_data in tracker.person_tracks.items():
        person_id = int(person_id)  # Convert NumPy int64 to Python int
        final_times[person_id] = {
            'original_time': float(track_data['last_seen'] - track_data['first_seen']),
            'corrected_time': float(track_data['last_seen'] - track_data['first_seen']),
            'corrections_applied': []
        }
    
    for correction in corrections:
        corrected_times = correction['agent_correction'].get('corrected_times', {})
        decision = correction['agent_correction'].get('decision', 'separate')
        
        for person_id_str, corrected_time in corrected_times.items():
            person_id = int(person_id_str)  # Ensure it's Python int
            
            if person_id in final_times:
                final_times[person_id]['corrected_time'] = float(corrected_time)
                final_times[person_id]['corrections_applied'].append({
                    'type': correction['confusion_event']['event_type'],
                    'decision': decision,
                    'confidence': float(correction['agent_correction'].get('confidence', 0))
                })
    
    os.makedirs("results", exist_ok=True)
    
    # ✅ FIX: Ensure all keys are Python int, not NumPy int64
    final_times_clean = {int(k): v for k, v in final_times.items()}
    
    with open("results/final_corrected_times.json", "w") as f:
        json.dump({
            'final_times': final_times_clean,
            'corrections': corrections
        }, f, indent=2)
    
    # Print summary
    print(f"\n{'='*60}")
    print("📊 FINAL RESULTS")
    print(f"{'='*60}")
    print(f"\nCustomer Times (Corrected):")
    
    for person_id, data in sorted(final_times_clean.items()):
        original = data['original_time']
        corrected = data['corrected_time']
        diff = corrected - original
        
        if len(data['corrections_applied']) > 0:
            print(f"\n  Customer {person_id}:")
            print(f"    Original:  {original:.1f}s")
            print(f"    Corrected: {corrected:.1f}s (Δ {diff:+.1f}s)")
            print(f"    Corrections: {len(data['corrections_applied'])}")
        else:
            print(f"\n  Customer {person_id}: {corrected:.1f}s (no corrections)")
    
    print(f"\n{'='*60}")
    print(f"✅ Complete! Results saved to results/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
