# OpenTelemetry test harness — span processors and handle hygiene

General notes for writing tests that create spans and assert on exported output.
Applicable whenever the codebase uses OpenTelemetry with a custom debug/JSONL exporter
and a `shutdown_provider()`-style teardown helper.

## Two processor families, two flush behaviors

| Processor | Export timing | Typical role in tests |
|-----------|--------------|-----------------------|
| `SimpleSpanProcessor` | Synchronous — `on_end()` pushes the span straight to the exporter | JSONL/debug exporters, any test that reads file content right after the call |
| `BatchSpanProcessor` | Asynchronous — spans sit in an internal queue, a worker thread flushes on a timer or at shutdown | Console or network exporters in production paths |

**Rule of thumb:** when the test asserts on **what was written**, attach the exporter with
`SimpleSpanProcessor`. A batched processor can lose queued spans if shutdown races the
worker thread, and the exact race behavior varies by SDK version.

## Why a test can "pass in isolation but fail in the suite"

Two common mechanisms, both worth checking before touching the assertions:

1. **Double file handle to the same path.**
   If `init_tracing()` creates a `JsonlSpanExporter` writing to the patched path and the
   test then creates **another** exporter writing to the same path (and overwrites the
   module's `_DEFAULT_EXPORTER`), two handles to the same file are open. On teardown one
   handle gets closed and the other stays open; on Windows that can block
   `tempfile.TemporaryDirectory` cleanup with a sharing violation, and can also confuse
   reads if one handle's buffered data hasn't landed when the other is closed.

   **Fix:** pick one owner for the file. Either let `init_tracing` own it (don't add
   another exporter in the test) or build the provider yourself and don't call
   `init_tracing` at all for that test.

2. **"Empty file" that is really a read-timing artifact.**
   Before changing assertions, reproduce with a one-off script that prints the file
   content both before and after teardown. If the content is present before teardown and
   gone after, the bug is in shutdown/close, not in the span emission. If the content was
   never there, the bug is upstream (wrong path, wrong provider, wrong processor).

## Two valid test setups

### A. Patch the path, let `init_tracing` own the exporter

```python
with patch.object(tracing, "_DEFAULT_JSONL", p):
    tracing.init_tracing("test-svc")   # one JsonlSpanExporter → p
    # ... call traced functions ...
    tracing.shutdown_provider()        # closes that exporter's handle
```

Do **not** create a second `JsonlSpanExporter` here. `init_tracing` already created one
that writes to the patched path.

### B. Build the provider manually

```python
prov = TracerProvider(resource=Resource.create({"service.name": "test"}))
exporter = JsonlSpanExporter(path=p)
prov.add_span_processor(SimpleSpanProcessor(exporter))
trace.set_tracer_provider(prov)
# ... call traced functions or start spans manually ...
exporter.shutdown()
# do not call tracing.shutdown_provider() — you own the provider
```

This avoids `init_tracing` entirely, so there is no second exporter and no singleton
overwrite. Useful when the test needs a clean provider or a non-default service name.

## Verifying shutdown behavior without guessing

When teardown looks wrong, read the installed SDK source rather than assuming.

```python
import inspect
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, BatchSpanProcessor

print(inspect.getsource(SimpleSpanProcessor.shutdown))
print(inspect.getsource(BatchSpanProcessor.shutdown))
```

Also useful: `inspect.getsource` on the underlying batch worker class if the processor
delegate is named something like `_batch_processor`. The goal is to see whether shutdown
flushes the queue before closing the exporter in the **installed** version, not the
version documented upstream.

## Default-argument patching note

`patch.object(module, "VAR", new_value)` works when the code under test reads
`VAR` as a **module attribute at call time**. It does **not** affect a default argument
that captured the original value at function-definition time. If a test changes
`module.VAR` and the code under test still sees the old value, check whether the code
uses `def f(x=module.VAR)` (captured at define time) versus `x = module.VAR` executed at
call time.
