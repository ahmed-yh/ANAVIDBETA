"""
The single CLI entry point: track customers, detect confusion events, and
correct dwell times with the Gemini agent. Used directly from the terminal
and by streamlit_app.py (launched as a subprocess in its own console window,
which is why it pauses on exit by default).

All arguments are optional and fall back to .env (via Config) when omitted,
so `python run_tracking.py` alone is a valid, fully headless-from-args run.
"""

import argparse
import json
import logging
import os
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from pipeline import run_tracking_and_analysis, parse_person_id

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Track customers, detect confusion events, and correct dwell times with AI."
    )
    parser.add_argument("video_path", nargs="?", default=None,
                         help="Path to input video (default: VIDEO_PATH from .env)")
    parser.add_argument("fps_limit", nargs="?", type=int, default=None,
                         help="Max FPS to process (default: FPS_LIMIT from .env)")
    parser.add_argument("disappear_threshold", nargs="?", type=float, default=None,
                         help="Seconds before considering a person exited (default: DISAPPEAR_THRESHOLD from .env)")
    parser.add_argument("--no-preview", action="store_true",
                         help="Don't open the OpenCV live-tracking window")
    parser.add_argument("--no-pause", action="store_true",
                         help="Don't wait for Enter at the end (for scripted/headless runs)")
    return parser.parse_args()


def main():
    """Run the complete queue intelligence system with AI analysis"""
    args = parse_args()

    video_path = args.video_path or Config.VIDEO_PATH
    fps_limit = args.fps_limit or Config.FPS_LIMIT
    disappear_threshold = args.disappear_threshold or Config.DISAPPEAR_THRESHOLD
    show_preview = not args.no_preview

    print("\n" + "="*60)
    print("🏪 QUEUE INTELLIGENCE SYSTEM - FULL AI ANALYSIS")
    print("="*60)
    print(f"Video: {video_path}")
    print(f"FPS Limit: {fps_limit}")
    print(f"Disappear Threshold: {disappear_threshold}s")
    print("="*60 + "\n")

    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        sys.exit(1)

    if not Config.validate():
        print("\n❌ Configuration errors - please fix .env file")
        print("Required: GOOGLE_API_KEY for AI analysis")
        sys.exit(1)

    worker_zones = Config.get_worker_zones()
    if worker_zones:
        print(f"✅ Worker zones configured: {len(worker_zones)} zone(s)")
    else:
        print("⚠️ No worker zones configured")

    print("\n🚀 Starting tracking...")
    if show_preview:
        print("📺 OpenCV window will open showing live tracking")
        print("Press 'q' in the OpenCV window to stop early\n")

    tracker, corrections = run_tracking_and_analysis(
        video_path,
        fps_limit=fps_limit,
        disappear_threshold=disappear_threshold,
        show_preview=show_preview,
    )

    os.makedirs("results", exist_ok=True)
    tracker.save_results("results/initial_tracking.csv")
    tracker.save_confusion_report("results/confusion_events.json")

    print(f"\n✅ Initial tracking complete")
    print(f"   Customers: {len(tracker.person_first_seen)}")
    print(f"   Confusions detected: {len(tracker.confusion_events)}")

    if not tracker.confusion_events:
        print("\n🎉 No confusions detected! All times are accurate.")
    else:
        print(f"\n" + "="*60)
        print(f"💾 Applying AI corrections")
        print("="*60)

    # final_corrected_times.json is always written, even with 0 confusions,
    # so downstream consumers (the Streamlit UI, build_demo_data.py) never
    # have to guess whether a leftover file is stale or genuinely empty.
    final_times = {}
    for person_id, track_data in tracker.person_tracks.items():
        person_id_int = int(person_id)
        final_times[person_id_int] = {
            'original_time': float(track_data['last_seen'] - track_data['first_seen']),
            'corrected_time': float(track_data['last_seen'] - track_data['first_seen']),
            'corrections_applied': []
        }

    for correction in corrections:
        analysis = correction['ai_analysis']
        same_person = analysis.get('same_person', False)
        decision = "MERGE" if same_person else "SEPARATE"
        reasoning = analysis.get('visual_evidence', 'No evidence provided')
        confidence = analysis.get('confidence', 0.0)

        for person_id_key, corrected_time in analysis.get('corrected_times', {}).items():
            if corrected_time is None:
                continue

            pid = parse_person_id(person_id_key)
            if pid is None:
                continue

            try:
                corrected_time_float = float(corrected_time)
            except (TypeError, ValueError):
                print(f"   ⚠️  Warning: Invalid corrected_time for ID {pid}: {corrected_time}")
                continue

            if pid not in final_times:
                continue

            if same_person and decision == 'MERGE':
                final_times[pid]['corrected_time'] = corrected_time_float
            elif not same_person and corrected_time_float > 0:
                final_times[pid]['corrected_time'] = corrected_time_float

            final_times[pid]['corrections_applied'].append({
                'type': correction['confusion_event']['event_type'],
                'decision': decision,
                'same_person': same_person,
                'confidence': confidence,
                'reasoning': reasoning,
            })

    with open("results/final_corrected_times.json", "w") as f:
        json.dump({
            'final_times': final_times,
            'corrections': corrections,
            'ai_analysis_log': [
                {
                    'confusion_id': i + 1,
                    'event_type': c['confusion_event']['event_type'],
                    'person_id': int(c['confusion_event']['person_id']),
                    'timestamp': float(c['confusion_event']['timestamp']),
                    'ai_decision': "MERGE" if c['ai_analysis'].get('same_person') else "SEPARATE",
                    'same_person': c['ai_analysis'].get('same_person', False),
                    'ai_confidence': float(c['ai_analysis'].get('confidence', 0.0)),
                    'ai_reasoning': c['ai_analysis'].get('visual_evidence', ''),
                    'visual_evidence': c['ai_analysis'].get('visual_evidence', ''),
                    'corrected_times': c['ai_analysis'].get('corrected_times', {}),
                }
                for i, c in enumerate(corrections)
            ]
        }, f, indent=2, cls=NumpyEncoder)

    print(f"\n" + "="*60)
    print("📊 FINAL RESULTS WITH AI CORRECTIONS")
    print("="*60)
    print(f"\nCustomer Times (AI-Corrected):")

    for person_id, data in final_times.items():
        original = data['original_time']
        corrected = data['corrected_time']
        diff = corrected - original

        if data['corrections_applied']:
            print(f"\n  Customer {person_id}:")
            print(f"    Original:  {original:.1f}s")
            print(f"    Corrected: {corrected:.1f}s (Δ {diff:+.1f}s)")
            print(f"    AI Corrections Applied: {len(data['corrections_applied'])}")
            for corr in data['corrections_applied']:
                same_person_str = "Same Person" if corr.get('same_person', False) else "Different People"
                print(f"      - {corr['type']}: {corr['decision']} ({same_person_str}, confidence: {corr['confidence']:.1%})")
        else:
            print(f"\n  Customer {person_id}: {corrected:.1f}s (no corrections needed)")

    print(f"\n" + "="*60)
    print(f"✅ Complete! All results saved to results/")
    print(f"   - initial_tracking.csv: Raw tracking data")
    print(f"   - confusion_events.json: Detected confusions")
    print(f"   - final_corrected_times.json: AI-corrected times with full analysis logs")
    print("="*60)

    if not args.no_pause:
        input("\nPress Enter to close this window...")


if __name__ == "__main__":
    main()
