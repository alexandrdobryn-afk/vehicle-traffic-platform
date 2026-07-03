from app.services import runtime_service


def _capabilities(cuda=False):
    return {
        "cpu": {"available": True},
        "cuda": {
            "available": cuda,
            "devices": ([{"index": 0, "name": "GPU", "memory_mb": 4096}] if cuda else []),
        },
        "torch_version": "test",
        "onnxruntime_version": "test",
        "onnxruntime_providers": [],
        "container_runtime": "test",
    }


def test_auto_uses_cuda_when_available(monkeypatch):
    monkeypatch.setattr(runtime_service, "get_runtime_capabilities", lambda: _capabilities(True))
    result = runtime_service.resolve_execution_policy("auto", 0, "fail_closed")
    assert result["resolved"] == "cuda"
    assert result["device"] == "cuda:0"


def test_auto_uses_cpu_when_cuda_is_unavailable(monkeypatch):
    monkeypatch.setattr(runtime_service, "get_runtime_capabilities", lambda: _capabilities(False))
    result = runtime_service.resolve_execution_policy("auto", 0, "fail_closed")
    assert result["resolved"] == "cpu"
    assert result["fallback_reason"] is None


def test_explicit_cuda_fails_closed(monkeypatch):
    monkeypatch.setattr(runtime_service, "get_runtime_capabilities", lambda: _capabilities(False))
    result = runtime_service.resolve_execution_policy("cuda", 0, "fail_closed")
    assert result["available"] is False
    assert result["resolved"] is None


def test_explicit_cuda_fallback_is_disclosed(monkeypatch):
    monkeypatch.setattr(runtime_service, "get_runtime_capabilities", lambda: _capabilities(False))
    result = runtime_service.resolve_execution_policy("cuda", 0, "allow_cpu")
    assert result["resolved"] == "cpu"
    assert result["fallback_reason"] == "requested_cuda_is_unavailable"
