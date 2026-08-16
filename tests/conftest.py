import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def tracker():
    """
    A DwellTimeTracker with the YOLO model mocked out, so tests exercise the
    real tracking/confusion-detection logic without loading model weights.
    """
    from queue_tracker import DwellTimeTracker

    with patch("queue_tracker.YOLO") as mock_yolo:
        mock_yolo.return_value = None
        t = DwellTimeTracker(disappear_threshold=10.0)
    return t
