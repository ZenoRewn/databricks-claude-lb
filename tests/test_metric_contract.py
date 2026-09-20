"""The exported API must remain consumable by its versioned contract."""
import json
from pathlib import Path
import unittest

from operations.metrics_contract import CONTRACT,parse_exposition
from request_telemetry import RequestTelemetry
from admission import AdmissionController
from usage_store import UsageDataStore
from cleanup_observability import CleanupTracker
from upstream_body import render_metrics


class MetricContractTests(unittest.TestCase):
    def test_all_runtime_families_parse_without_ids_or_label_loss(self):
        telemetry=RequestTelemetry();telemetry.send_results[('copilot','responses','http_2xx')]=1
        admission=AdmissionController(max_active=1,max_queued=1,wait_timeout=1,body_budget=100,tenant_limits={})
        text=telemetry.render()+admission.render_metrics()+UsageDataStore().render_metrics()+CleanupTracker().render()+render_metrics()
        declared={line.split()[2] for line in text.splitlines() if line.startswith('# TYPE ')}
        self.assertEqual(declared,set(CONTRACT['families']))
        series=parse_exposition(text)
        self.assertTrue(series)
        self.assertTrue(all('request_id' not in value['labels'] for value in series))
        self.assertTrue(any(value['labels'].get('outcome')=='failed' for value in series))
        saved=json.loads((Path(__file__).parents[1]/'operations/metrics-contract.json').read_text())
        self.assertEqual(saved,CONTRACT)
