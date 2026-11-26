"""Installing metrics: the SDK meter provider, where metrics go, and the views in front.

The composition root calls `configure_metrics_from_settings` once at process
start (the API's lifespan, the worker's `main`), beside tracing. As with
tracing, configuration alone decides what is installed:

- `METRICS_ENABLED=false` (the default) installs nothing; every instrument in
  `personalos.observability.metrics` is then a no-op.
- `OTEL_EXPORTER_OTLP_ENDPOINT` names the OTLP/HTTP collector metrics are
  pushed to every `METRICS_EXPORT_INTERVAL_SECONDS`, the same endpoint spans
  go to. The compose stack's collector turns them into a Prometheus scrape
  target (`deploy/otel/collector.yaml`).

Every instrument gets a view that keeps only the labels its catalog entry
declares (`MetricSpec.labels`) -- the label allow-list enforced a second time,
inside the SDK, for anything recorded around `metrics.add`/`metrics.observe`
-- and every histogram gets the bucket boundaries its entry names, since the
SDK's defaults are sized for milliseconds.
"""

import logging
from typing import Any

from opentelemetry import metrics as otel_metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import (
    MetricExporter,
    MetricReader,
    PeriodicExportingMetricReader,
)
from opentelemetry.sdk.metrics.view import ExplicitBucketHistogramAggregation, View
from opentelemetry.sdk.resources import Resource

from personalos.config import Settings, settings
from personalos.observability import metrics

logger = logging.getLogger(__name__)


def catalog_views() -> list[View]:
    """One view per catalog metric: its labels only, and its histogram buckets."""
    views = []
    for spec in metrics.CATALOG:
        kwargs: dict[str, Any] = {
            "instrument_name": spec.name,
            "attribute_keys": set(spec.labels),
        }
        if spec.kind is metrics.MetricKind.HISTOGRAM and spec.buckets:
            kwargs["aggregation"] = ExplicitBucketHistogramAggregation(boundaries=spec.buckets)
        views.append(View(**kwargs))
    return views


def _otlp_exporter(endpoint: str) -> MetricExporter | None:
    try:
        from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
    except ImportError:
        logger.warning(
            "OTEL_EXPORTER_OTLP_ENDPOINT is set but opentelemetry-exporter-otlp-proto-http "
            "is not installed; metrics will not be exported"
        )
        return None
    if not endpoint.rstrip("/").endswith("/v1/metrics"):
        endpoint = f"{endpoint.rstrip('/')}/v1/metrics"
    exporter: MetricExporter = OTLPMetricExporter(endpoint=endpoint)
    return exporter


def configure_metrics(
    *,
    enabled: bool = True,
    service_name: str = "personalos",
    otlp_endpoint: str | None = None,
    reader: MetricReader | None = None,
    export_interval_seconds: float = 15.0,
    install_global: bool = True,
    resource_attributes: dict[str, Any] | None = None,
) -> MeterProvider | None:
    """Install a meter provider for PersonalOS metrics; `None`, and nothing installed, if disabled.

    `reader` is for tests (an `InMemoryMetricReader`); otherwise
    `otlp_endpoint` gets a periodic OTLP exporter. With neither, instruments
    record into a provider nobody reads, which is what a dry run wants.
    `install_global` also makes the provider OpenTelemetry's global one,
    which can happen once per process; a test passes `False` and calls
    `reset_metrics` after.
    """
    if not enabled:
        return None

    readers: list[MetricReader] = []
    if reader is not None:
        readers.append(reader)
    elif otlp_endpoint:
        exporter = _otlp_exporter(otlp_endpoint)
        if exporter is not None:
            readers.append(
                PeriodicExportingMetricReader(
                    exporter, export_interval_millis=export_interval_seconds * 1000
                )
            )

    provider = MeterProvider(
        resource=Resource.create({"service.name": service_name, **(resource_attributes or {})}),
        metric_readers=readers,
        views=catalog_views(),
    )
    metrics.set_meter_provider(provider)
    if install_global:
        otel_metrics.set_meter_provider(provider)
    logger.info(
        "metrics enabled for %s (exporter: %s)",
        service_name,
        "custom" if reader is not None else (otlp_endpoint or "none"),
    )
    return provider


def configure_metrics_from_settings(
    service_name: str, config: Settings = settings
) -> MeterProvider | None:
    """Install metrics as the environment says: on or off, which endpoint, how often."""
    return configure_metrics(
        enabled=config.metrics_enabled,
        service_name=config.otel_service_name or service_name,
        otlp_endpoint=config.otel_exporter_otlp_endpoint or None,
        export_interval_seconds=config.metrics_export_interval_seconds,
        resource_attributes={"deployment.environment": config.app_env},
    )


def reset_metrics() -> None:
    """Stop creating PersonalOS instruments from a configured provider."""
    provider = metrics.meter_provider()
    metrics.set_meter_provider(None)
    if isinstance(provider, MeterProvider):
        provider.shutdown()


__all__ = [
    "catalog_views",
    "configure_metrics",
    "configure_metrics_from_settings",
    "reset_metrics",
]
