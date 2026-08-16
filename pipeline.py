"""
Shared tracking + AI-analysis loop used by both CLI entry points
(run_system.py and run_tracking_terminal.py), which previously duplicated
this ~80 lines of logic and had drifted into inconsistent person-ID parsing.
Each entry point still owns its own output format/filenames and interactivity.
"""

import logging
from typing import Optional

from config import Config
from queue_tracker import DwellTimeTracker
from tools.segment_extractor import extract_segment_with_context
from agent import create_segment_analyzer_agent, analyze_confusion_segment

logger = logging.getLogger(__name__)


def parse_person_id(key) -> Optional[int]:
    """
    Extract the base numeric person id from an AI correction key.
    Keys look like "45" or "45_visit1" (from the return_after_leave prompt).
    """
    base = str(key).split('_')[0]
    try:
        return int(base)
    except ValueError:
        return None


def run_tracking_and_analysis(
    video_path: str,
    fps_limit: Optional[float] = None,
    disappear_threshold: Optional[float] = None,
    show_preview: bool = True,
    segment_padding_seconds: float = 5.0,
):
    """
    Track all people in the video, detect confusion events, and analyze each
    one with the Gemini agent.

    Returns:
        (tracker, corrections) where corrections is a list of
        {'confusion_event': dict, 'ai_analysis': dict, 'segment_path': str}
    """
    fps_limit = fps_limit or Config.FPS_LIMIT
    disappear_threshold = disappear_threshold or Config.DISAPPEAR_THRESHOLD
    worker_zones = Config.get_worker_zones()

    tracker = DwellTimeTracker(
        model_path=Config.YOLO_MODEL,
        disappear_threshold=disappear_threshold,
        fps_limit=fps_limit,
        exclude_zones=worker_zones,
    )

    tracker.process_video(
        video_path,
        output_path=Config.OUTPUT_VIDEO_PATH,
        fps_limit=fps_limit,
        show_preview=show_preview,
    )

    corrections = []
    if not tracker.confusion_events:
        return tracker, corrections

    agent = create_segment_analyzer_agent()

    for i, confusion_event in enumerate(tracker.confusion_events, 1):
        logger.info(
            "Analyzing confusion %d/%d (%s)",
            i, len(tracker.confusion_events), confusion_event.event_type,
        )

        segment_context = extract_segment_with_context(
            video_path=video_path,
            confusion_event=confusion_event.to_dict(),
            person_tracks=tracker.person_tracks,
            padding_seconds=segment_padding_seconds,
            output_dir="data/output/segments",
        )

        try:
            analysis = analyze_confusion_segment(segment_context, agent)
        except Exception:
            logger.exception(
                "AI analysis failed for confusion %d/%d", i, len(tracker.confusion_events)
            )
            continue

        corrections.append({
            'confusion_event': confusion_event.to_dict(),
            'ai_analysis': analysis,
            'segment_path': segment_context['segment_path'],
        })

    return tracker, corrections
