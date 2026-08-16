import numpy as np


def _box(x1, y1, x2, y2):
    return np.array([x1, y1, x2, y2], dtype=float)


def test_id_switch_confirmed_after_debounce_frames(tracker):
    """
    A regression test for a bug where the id-switch candidate counter was
    wiped in the same call it was created, so id_switch events could never
    fire no matter how long the switch persisted.
    """
    # Frame 0: person tracked as ID 1
    events = tracker.detect_confusion([1], [1], [_box(100, 100, 150, 200)], [0.9], 0.0, 0)
    assert events == []

    # Frame 1: ID 1 vanishes, ID 2 appears right next to where it was (a switch candidate)
    events = tracker.detect_confusion([2], [2], [_box(102, 101, 152, 201)], [0.9], 0.1, 1)
    assert events == []
    assert len(tracker.id_switch_candidates) == 1

    # Frames 2-4: ID 2 keeps being tracked, ID 1 never reappears
    for frame in range(2, tracker.confusion_debounce_frames):
        events = tracker.detect_confusion([2], [2], [_box(103, 102, 153, 202)], [0.9], frame * 0.1, frame)
        assert events == []

    # On the confusion_debounce_frames'th confirming frame, the switch should fire
    events = tracker.detect_confusion(
        [2], [2], [_box(103, 102, 153, 202)], [0.9],
        tracker.confusion_debounce_frames * 0.1, tracker.confusion_debounce_frames,
    )
    assert len(events) == 1
    assert events[0].event_type == 'id_switch'
    assert events[0].context['old_id'] == 1
    assert events[0].context['new_id'] == 2
    # Candidate should be resolved (removed) once reported
    assert tracker.id_switch_candidates == {}


def test_id_switch_candidate_discarded_if_old_id_reappears(tracker):
    tracker.detect_confusion([1], [1], [_box(100, 100, 150, 200)], [0.9], 0.0, 0)
    tracker.detect_confusion([2], [2], [_box(102, 101, 152, 201)], [0.9], 0.1, 1)
    assert len(tracker.id_switch_candidates) == 1

    # ID 1 reappears alongside ID 2 -> this wasn't a real switch, candidate should drop
    events = tracker.detect_confusion([1, 2], [1, 2], [_box(100, 100, 150, 200), _box(103, 102, 153, 202)], [0.9, 0.9], 0.2, 2)
    assert events == []
    assert tracker.id_switch_candidates == {}


def test_id_switch_not_flagged_when_far_apart(tracker):
    tracker.detect_confusion([1], [1], [_box(0, 0, 50, 100)], [0.9], 0.0, 0)
    # New ID appears far away -> shouldn't even become a candidate
    events = tracker.detect_confusion([2], [2], [_box(900, 900, 950, 1000)], [0.9], 0.1, 1)
    assert events == []
    assert tracker.id_switch_candidates == {}


def test_occlusion_detected_after_brief_disappearance(tracker):
    tracker.person_disappeared[7] = 1.0
    events = tracker.detect_confusion([7], [7], [_box(0, 0, 10, 10)], [0.9], current_time=4.0, frame_count=10)
    occlusions = [e for e in events if e.event_type == 'occlusion']
    assert len(occlusions) == 1
    assert occlusions[0].person_id == 7


def test_return_after_leave_detected_beyond_threshold(tracker):
    tracker.person_disappeared[7] = 0.0
    events = tracker.detect_confusion(
        [7], [7], [_box(0, 0, 10, 10)], [0.9],
        current_time=tracker.disappear_threshold + 1.0, frame_count=10,
    )
    returns = [e for e in events if e.event_type == 'return_after_leave']
    assert len(returns) == 1
    assert returns[0].person_id == 7
