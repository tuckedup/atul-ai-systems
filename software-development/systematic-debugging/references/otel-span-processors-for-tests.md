# OpenTelemetry Span Processors for Tests

## Two processors, two export timings

| Processor | Export timing | Use for |
|-----------|--------------|---------|
| `SimpleSpanProcessor` | Immediate — exports each span the moment `on_end()` fires | Debugging, tests, low-throughput JSONL/audit |
| `BatchSpanProcessor` | Deferred — spans queued, flushed on a timer (default ~500ms) or on shutdown | Production high-throughput paths |

**Rule:** in tests that assert on span content, attach `SimpleSpanProcessor` to the exporter. A `BatchSpanProcessor` may drop queued spans if `shutdown()` is called before the timer fires (the worker thread may not have flushed the queue).

## Proof from this workspace (OTel SDK 1.44.0)

`SimpleSpanProcessor.on_end()` calls `self.span_exporter.export((span,))` synchronously, then `shutdown()` calls `self.span_exporter.shutdown()`. No timer, no queue, no race.

`BatchSpanProcessor.on_end()` calls `self._batch_processor.emit(span)` which appends to an internal deque. The worker thread reads the deque on a timer. On `shutdown()`, the worker is woken, exports remaining items, then the exporter is shut down. **This works correctly in 1.44.0** — the worker exports `EXPORT_ALL` after breaking out of its loop — but it is SDK-version-specific and should not be assumed.

## What `init_tracing()` wires (this codebase)

```python
# aisys/tracing.py::init_tracing()
provider = TracerProvider(...)
provider.add_span_processor(BatchSpanProcessor(console_exporter))   # Console — batched
provider.add_span_processor(SimpleSpanProcessor(jsonl_exporter))    # JSONL — immediate
```

Console output is batched (fine for human reading). JSONL writes are immediate (fine for tests that read the file after each call).

## Test pattern that avoids double-exporter handle leaks

**Problem:** `init_tracing()` creates a `JsonlSpanExporter` and stores it in `_DEFAULT_EXPORTER`. If a test then creates a second `JsonlSpanExporter(path=p)` and overwrites `_DEFAULT_EXPORTER`, the first exporter's file handle is leaked. On Windows, this blocks `tempfile.TemporaryDirectory` cleanup with `PermissionError [WinError 32]`.

**Two valid test setups:**

### A. Let `init_tracing` own the exporter (preferred for path-patching tests)

```python
with patch.object(tracing, "_DEFAULT_JSONL", p):
    tracing.init_tracing("test-svc")   # creates ONE JsonlSpanExporter → p
    # ... call traced functions ...
    tracing.shutdown_provider()        # closes the one exporter's handle
```

Do NOT create a second `JsonlSpanExporter` in this setup. `init_tracing` already made one that writes to the patched path.

### B. Build the provider manually (preferred for full control)

```python
prov = TracerProvider(resource=Resource.create({"service.name": "test"}))
exporter = JsonlSpanExporter(path=p)
prov.add_span_processor(SimpleSpanProcessor(exporter))
trace.set_tracer_provider(prov)
# ... call traced functions or start spans manually ...
exporter.shutdown()
# no need to call tracing.shutdown_provider() — you own the provider
```

This avoids `init_tracing` entirely, so there is no second exporter and no singleton overwrite.

## `shutdown_provider()` contract

`tracing.shutdown_provider()` does:
1. `prov.force_flush()` — flushes any batch processors
2. `prov.shutdown()` — shuts down all span processors (which shut down their exporters)
3. `_close_default_jsonl()` — closes `_DEFAULT_EXPORTER._f` if it is still open, then sets `_DEFAULT_EXPORTER = None`
4. `DEBUG_SPANS.clear()`

Step 3 is a backstop for the case where an exporter was attached outside `init_tracing` (setup B above) and `prov.shutdown()` did not reach it. In setup A, step 2 already closed the handle via the `SimpleSpanProcessor` → `exporter.shutdown()` chain; step 3 is a no-op.

## Mock patching `_DEFAULT_JSONL`

`unittest.mock.patch.object(tracing, "_DEFAULT_JSONL", p)` works because `_DEFAULT_JSONL` is accessed as a module attribute **at call time** inside `JsonlSpanExporter.__init__` (`self._path = path or _DEFAULT_JSONL`). This is distinct from default-argument capture (see the `test-driven-development` pitfall on default-arg patching).
