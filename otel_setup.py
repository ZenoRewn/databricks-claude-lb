"""P3.1: OpenTelemetry tracing bootstrap for the LB.

Optional runtime dependency — if the opentelemetry-* packages are not
installed, `setup_tracing()` returns immediately (no-op) and the LB
continues without traces.

Env vars respected (all standard OTel):
- OTEL_ENABLED             — master switch (default `false`; opt-in to keep
                              startup fast when no collector is deployed)
- OTEL_SERVICE_NAME        — default `databricks-claude-lb`
- OTEL_EXPORTER_OTLP_ENDPOINT  — e.g. `http://otel-collector:4318`; when set
                              spans are exported over OTLP HTTP; when unset
                              spans stay in-process and are logged to stdout
                              via ConsoleSpanExporter (still useful for
                              local dev)
- OTEL_EXPORTER_OTLP_HEADERS
- OTEL_RESOURCE_ATTRIBUTES — e.g. `deployment.environment=prod,cluster=aks-1`

Auto-instruments:
- FastAPI (server-side HTTP spans, one per request)
- httpx AsyncClient (client-side spans for every Databricks/Azure/Copilot
  upstream call, nested under the server span)

Caller: `from otel_setup import setup_tracing; setup_tracing(app)` in
lifespan. Returns True if tracing was set up, False if disabled/unavailable.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger("main")


def _otel_enabled() -> bool:
    v = os.getenv("OTEL_ENABLED", "").strip().lower()
    return v in ("1", "true", "yes", "on")


def setup_tracing(app) -> bool:
    """Configure OpenTelemetry tracing if enabled + packages available.

    Returns True on success, False if opt-out or dependency missing.
    """
    if not _otel_enabled():
        return False

    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource, SERVICE_NAME
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import (
            BatchSpanProcessor,
            ConsoleSpanExporter,
        )
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
    except ImportError as e:
        logger.warning(
            "[OTel] OTEL_ENABLED=true but opentelemetry packages are missing (%s). "
            "Install: pip install opentelemetry-api opentelemetry-sdk "
            "opentelemetry-instrumentation-fastapi opentelemetry-instrumentation-httpx "
            "opentelemetry-exporter-otlp-proto-http. Tracing DISABLED.",
            type(e).__name__,
        )
        return False

    service_name = os.getenv("OTEL_SERVICE_NAME", "databricks-claude-lb")
    resource = Resource.create({SERVICE_NAME: service_name})
    provider = TracerProvider(resource=resource)

    otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip()
    if otlp_endpoint:
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                OTLPSpanExporter,
            )
            provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
            logger.info(
                "[OTel] tracing enabled, exporting to OTLP endpoint=%s service=%s",
                otlp_endpoint, service_name,
            )
        except ImportError:
            logger.warning(
                "[OTel] OTLP endpoint set but opentelemetry-exporter-otlp-proto-http "
                "missing — falling back to ConsoleSpanExporter",
            )
            provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    else:
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
        logger.info(
            "[OTel] tracing enabled, ConsoleSpanExporter (no OTLP_ENDPOINT set) "
            "service=%s",
            service_name,
        )

    trace.set_tracer_provider(provider)

    # FastAPI instrumentation — one server span per request. excluded_urls avoids
    # noise from health probes and /metrics scrapes; those are high-cardinality
    # low-value traffic.
    FastAPIInstrumentor.instrument_app(
        app,
        excluded_urls="health,health/live,health/ready,metrics",
    )
    # httpx auto-instrument — every Databricks/Azure/Copilot upstream call gets
    # a client span nested under the server span. This is the single most
    # valuable trace: shows exactly where per-request latency is spent.
    HTTPXClientInstrumentor().instrument()
    return True


def get_tracer():
    """Convenience: return the LB tracer (or a no-op tracer if setup wasn't run)."""
    try:
        from opentelemetry import trace
        return trace.get_tracer("databricks-claude-lb")
    except ImportError:
        # Return a shim with a no-op start_as_current_span context manager.
        import contextlib
        class _NoopTracer:
            @contextlib.contextmanager
            def start_as_current_span(self, name, **kwargs):
                yield None
        return _NoopTracer()
