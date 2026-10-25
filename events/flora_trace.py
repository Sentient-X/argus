"""Automatic Waddle spans around the audited Flora implementation.

Only the hosted process installs these scoped adapters. Preparation source bytes and
cache identity stay unchanged; telemetry never retries or changes a model request.
"""
from __future__ import annotations

from contextlib import contextmanager, ExitStack
from contextvars import copy_context
from functools import wraps
import hashlib
import json
import os
from pathlib import Path
import sys
import uuid

from .flora_journal import input_digest


def sdk_files(source: Path) -> dict[str, str]:
    """An explicit source-only SDK bundle; never ship checkout credentials or caches."""
    if not (source / "_run.py").is_file() or "def span(" not in (source / "_run.py").read_text():
        raise ValueError("FLORA_WADDLE_SOURCE must select the Waddle SDK with Run.span support")
    return {str(path.relative_to(source)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(source.rglob("*"))
            if path.is_file() and (path.suffix == ".py" or path.name == "py.typed")
            and "__pycache__" not in path.parts}


def sdk_source() -> tuple[Path, dict[str, str]]:
    selected = os.environ.get("FLORA_WADDLE_SOURCE")
    if not selected:
        raise ValueError("Set FLORA_WADDLE_SOURCE to the selected glued/autonomy/waddle/waddle SDK directory")
    source = Path(selected).resolve()
    files = sdk_files(source)
    expected = os.environ.get("FLORA_WADDLE_SHA256")
    if expected and input_digest(files) != expected:
        raise ValueError("Packaged Waddle SDK differs from its submitted SHA256")
    return source, files


class FloraTrace:
    def __init__(self, root: Path, manifest: dict, *, phase: str = "worker"):
        self.root, self.manifest, self.phase = root, manifest, phase
        self.run = None
        self.failures: list[str] = []
        self.receipt = {"run_id": uuid.uuid5(uuid.NAMESPACE_URL,
            f"argus-flora:{manifest['run_id']}:{phase}").hex,
            "flora_run_id": manifest["run_id"], "phase": phase,
            "manifest_sha256": input_digest(manifest), "status": "unavailable",
            "database": "waddle.duckdb", "source": "automatic"}

    def failed(self, exc: Exception) -> None:
        # Do not copy exception text that could contain hosted credentials.
        if len(self.failures) < 8:
            self.failures.append(type(exc).__name__)

    def retain(self) -> None:
        try:
            path = self.root / "trace.json"
            temporary = path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps({**self.receipt, "errors": self.failures}, sort_keys=True))
            temporary.replace(path)
        except Exception as exc:
            self.failed(exc)

    def __enter__(self):
        self.root.mkdir(parents=True, exist_ok=True)
        try:
            import waddle
            from waddle import RunType, WorkerInfo
            source = Path(waddle.__file__).parent
            files = sdk_files(source)
            digest = input_digest(files)
            expected = os.environ.get("FLORA_WADDLE_SHA256")
            if ((expected and digest != expected) or
                    self.manifest.get("telemetry_sdk_sha256", digest) != digest):
                raise ValueError("Waddle runtime source differs from the submitted SDK")
            self.receipt["sdk_sha256"] = digest
            self.receipt["sdk_files"] = files
            # No ambient cloud writes without a destination AND an authenticated grant.
            hosted = all(os.environ.get(key) for key in
                         ("SX_WADDLE_URL", "SX_PROJECT_ID", "SX_API_KEY"))
            self.run = waddle.init(name=f"Flora {self.manifest['run_id']} {self.phase}",
                run_id=self.receipt["run_id"], db_path=str(self.root / "waddle.duckdb"),
                config={"flora": self.receipt.copy(), "stage": self.manifest["stage"],
                        "implementation_sha256": self.manifest.get("implementation_sha256", {})},
                system_metrics=True, capture_logging=False, sync=None if hosted else False,
                worker=WorkerInfo(node_id=os.environ.get("MODAL_TASK_ID", "flora-local")),
                run_type=RunType.BENCHMARK)
            self.receipt["status"] = "running"
        except Exception as exc:
            self.failed(exc)
        self.retain()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.run is not None:
            try:
                self.run.__exit__(exc_type, exc, tb)
                # Checkpoint local spool bytes before the volume receipt hashes them.
                import duckdb
                with duckdb.connect(str(self.root / "waddle.duckdb")) as connection:
                    connection.execute("CHECKPOINT")
                self.receipt["status"] = "failed" if exc_type else "completed"
                self.receipt["span_record_failures"] = self.run.span_record_failures
            except Exception as error:
                self.failed(error)
                self.receipt["status"] = "incomplete"
        self.retain()

    @contextmanager
    def span(self, name: str):
        manager = None
        if self.run is not None:
            try:
                manager = self.run.span(name)
                manager.__enter__()
            except Exception as exc:
                self.failed(exc)
                manager = None
        try:
            yield
        except BaseException:
            if manager is not None:
                try:
                    manager.__exit__(*sys.exc_info())
                except BaseException as exc:
                    if isinstance(exc, Exception):
                        self.failed(exc)
            raise
        else:
            if manager is not None:
                try:
                    manager.__exit__(None, None, None)
                except Exception as exc:
                    self.failed(exc)

    @contextmanager
    def instrument(self):
        """Scoped to one dedicated worker, including its bounded preparation thread."""
        if self.run is None:
            yield
            return
        from . import flora_vllm, flora_journal, flora_hosted
        with ExitStack() as restore:
            def replace(owner, name, value):
                original = getattr(owner, name)
                try:
                    setattr(owner, name, value)
                except Exception as exc:
                    self.failed(exc)
                    return
                restore.callback(setattr, owner, name, original)

            def wrap(owner, name, label):
                original = getattr(owner, name)
                @wraps(original)
                def traced(*args, **kwargs):
                    with self.span(label):
                        return original(*args, **kwargs)
                replace(owner, name, traced)

            for name, label in (("_prepare_caption_many", "input.prepare"),
                                ("_load_prepared_cache", "input.cache_read"),
                                ("prepare_many_cache", "input.cache_write"),
                                ("prepare", "input.native_prepare"),
                                ("_batch", "caption.batch")):
                wrap(flora_vllm, name, label)
            wrap(flora_journal, "_save", "output.journal_write")
            wrap(flora_hosted, "save", "output.persist")
            original_load = flora_vllm.load_engine
            @wraps(original_load)
            def load(*args, **kwargs):
                with self.span("model.load"):
                    engine = original_load(*args, **kwargs)
                wrap(engine, "generate", "model.generate")
                return engine
            replace(flora_vllm, "load_engine", load)
            original_executor = flora_vllm.ThreadPoolExecutor
            tracer = self
            class TracedExecutor(original_executor):
                def submit(self, fn, /, *args, **kwargs):
                    context = copy_context()
                    future = super().submit(context.run, fn, *args, **kwargs)
                    original_result = future.result
                    @wraps(original_result)
                    def result(*result_args, **result_kwargs):
                        with tracer.span("input.wait"):
                            return original_result(*result_args, **result_kwargs)
                    future.result = result
                    return future
            replace(flora_vllm, "ThreadPoolExecutor", TracedExecutor)
            yield
