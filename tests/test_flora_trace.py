"""Producer boundary, including best-effort telemetry failures; no paid inference."""
import json
import sys
from types import SimpleNamespace

import pytest

from events.flora_trace import FloraTrace, sdk_files, sdk_source


def test_sdk_bundle_is_explicit_and_source_only(tmp_path, monkeypatch):
    monkeypatch.delenv("FLORA_WADDLE_SOURCE", raising=False)
    with pytest.raises(ValueError, match="Set FLORA_WADDLE_SOURCE"):
        sdk_source()
    (tmp_path / "_run.py").write_text("def span(self): pass")
    (tmp_path / ".env").write_text("never package me")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "untrusted.py").write_text("no")
    assert set(sdk_files(tmp_path)) == {"_run.py"}
    monkeypatch.setenv("FLORA_WADDLE_SOURCE", str(tmp_path))
    monkeypatch.setenv("FLORA_WADDLE_SHA256", "bad")
    with pytest.raises(ValueError, match="SHA256"):
        sdk_source()


def test_missing_sdk_is_retained_and_never_changes_the_work(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "waddle", SimpleNamespace())
    with FloraTrace(tmp_path, {"run_id": "unavailable", "stage": "caption"}) as trace:
        with trace.instrument(), trace.span("model.generate"):
            output = "original result"
    assert output == "original result"
    receipt = json.loads((tmp_path / "trace.json").read_text())
    assert receipt["status"] == "unavailable" and receipt["errors"]


def test_trace_failure_preserves_original_model_exception(tmp_path):
    trace = FloraTrace(tmp_path, {"run_id": "failure", "stage": "caption"})
    class BrokenSpan:
        def __enter__(self):
            pass
        def __exit__(self, *args):
            raise RuntimeError("telemetry failed")
    trace.run = SimpleNamespace(span=lambda name: BrokenSpan())
    original = ValueError("model failed")
    with pytest.raises(ValueError) as caught:
        with trace.span("model.generate"):
            raise original
    assert caught.value is original
    assert trace.failures == ["RuntimeError"]
