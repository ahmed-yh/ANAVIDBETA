import torch

from gpu_utils import resolve_device, describe_device


def test_resolve_device_returns_cpu_when_cuda_unavailable(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device() == 'cpu'


def test_resolve_device_returns_gpu_index_when_cuda_available(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda i: "Fake GPU")
    monkeypatch.setattr(torch.version, "cuda", "12.1")
    assert resolve_device() == 0


def test_describe_device_cpu():
    assert describe_device('cpu') == "CPU"


def test_describe_device_gpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda i: "Fake GPU")
    assert "Fake GPU" in describe_device(0)
