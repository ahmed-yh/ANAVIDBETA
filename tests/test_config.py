import json

from config import Config


def test_get_worker_zones_parses_valid_json(monkeypatch):
    zones = [[[0, 0], [10, 0], [10, 10], [0, 10]]]
    monkeypatch.setenv("WORKER_ZONES", json.dumps(zones))
    result = Config.get_worker_zones()
    assert result == [[(0, 0), (10, 0), (10, 10), (0, 10)]]


def test_get_worker_zones_falls_back_on_invalid_json(monkeypatch):
    monkeypatch.setenv("WORKER_ZONES", "{not valid json")
    assert Config.get_worker_zones() == []


def test_get_worker_zones_defaults_to_empty(monkeypatch):
    monkeypatch.delenv("WORKER_ZONES", raising=False)
    assert Config.get_worker_zones() == []


def test_validate_fails_without_api_key(monkeypatch, tmp_path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"fake")

    monkeypatch.setattr(Config, "GOOGLE_API_KEY", "")
    monkeypatch.setattr(Config, "VIDEO_PATH", str(video))
    assert Config.validate() is False


def test_validate_fails_when_video_missing(monkeypatch):
    monkeypatch.setattr(Config, "GOOGLE_API_KEY", "fake-key")
    monkeypatch.setattr(Config, "VIDEO_PATH", "does/not/exist.mp4")
    assert Config.validate() is False


def test_validate_passes_with_key_and_video(monkeypatch, tmp_path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"fake")

    monkeypatch.setattr(Config, "GOOGLE_API_KEY", "fake-key")
    monkeypatch.setattr(Config, "VIDEO_PATH", str(video))
    assert Config.validate() is True
