"""Deterministic window and archive receipts, using synthetic snapshots only."""
import json
from pathlib import Path
import tempfile
import unittest

from operations.reporting import calculate_window, archive_report, verify_archive, record_delivery


class WindowTests(unittest.TestCase):
    def snapshots(self):
        return [{'timestamp':t,'pod_uid':'pod-a','container_start_time':'2026-09-18T00:00:00Z',
                 'collection_status':'ok','metrics':{'requests':v,'id_fields_stripped':v*2}}
                for t,v in [('2026-09-19T01:30:12Z',10),('2026-09-19T02:00:00Z',14),('2026-09-19T02:30:12Z',19)]]
    def calc(self,samples,**kwargs):
        return calculate_window(samples,'2026-09-19T01:30:12Z','2026-09-19T02:30:12Z',
                                {'requests':'endpoint_admissions','id_fields_stripped':'fields'},**kwargs)
    def test_exact_window_and_units(self):
        result=self.calc(self.snapshots())
        self.assertEqual(result['metrics']['requests']['observed_delta'],9)
        self.assertEqual(result['metrics']['requests']['status'],'complete')
        self.assertEqual(result['metrics']['id_fields_stripped']['unit'],'fields')
        self.assertEqual(result['actual_first_sample'],'2026-09-19T01:30:12+00:00')
    def test_missing_anchor_is_partial_not_a_full_window(self):
        result=self.calc(self.snapshots()[1:])
        self.assertEqual(result['metrics']['requests']['observed_delta'],5)
        self.assertEqual(result['metrics']['requests']['status'],'partial')
        self.assertGreater(result['head_gap_seconds'],0)
    def test_pod_replacement_and_same_pod_reset_are_not_zero(self):
        for replacement in (True,False):
            with self.subTest(replacement=replacement):
                samples=self.snapshots()
                if replacement:samples[-1]['pod_uid']='pod-b'
                else:samples[-1]['metrics']['requests']=1
                result=self.calc(samples)
                self.assertEqual(result['metrics']['requests']['observed_delta'],4)
                self.assertEqual(result['metrics']['requests']['status'],'partial')
                self.assertTrue(result['metrics']['requests']['excluded_intervals'])
    def test_missing_metric_is_unknown_and_missing_collection_is_not_skipped(self):
        samples=self.snapshots()
        for sample in samples:sample['metrics'].pop('requests')
        result=self.calc(samples)
        self.assertIsNone(result['metrics']['requests']['observed_delta'])
        self.assertEqual(result['metrics']['requests']['status'],'unknown')
        samples=self.snapshots();samples[1]['collection_status']='failed'
        self.assertIsNone(self.calc(samples)['metrics']['requests']['observed_delta'])
    def test_outside_anchor_is_explicit_and_not_exact_nominal_coverage(self):
        samples=self.snapshots();samples[0]['timestamp']='2026-09-19T01:30:00Z'
        result=self.calc(samples,anchor_policy='include_previous')
        self.assertEqual(result['metrics']['requests']['observed_delta'],9)
        self.assertEqual(result['anchor_extension_seconds'],12)
        self.assertEqual(result['metrics']['requests']['status'],'partial')
    def test_long_gap_and_naive_timestamps_do_not_become_healthy(self):
        result=self.calc(self.snapshots(),max_gap_seconds=100)
        self.assertEqual(result['max_gap_seconds'],100)
        self.assertIsNone(result['metrics']['requests']['observed_delta'])
        with self.assertRaises(ValueError):
            calculate_window([], '2026-09-19T01:30:00','2026-09-19T02:30:00Z',{'requests':'requests'})
    def test_duplicate_sample_and_invalid_counter_are_rejected_or_unknown(self):
        samples=self.snapshots()
        with self.assertRaises(ValueError):self.calc(samples+[samples[0]])
        samples[1]['metrics']['requests']=float('nan')
        self.assertEqual(self.calc(samples)['metrics']['requests']['status'],'unknown')


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.summary=WindowTests().calc(WindowTests().snapshots())
    def test_archive_reads_back_and_delivery_has_an_independent_state(self):
        receipt=archive_report(self.root,'run-1',self.summary)
        self.assertEqual(receipt['execution_status'],'succeeded')
        self.assertEqual(receipt['artifact_status'],'verified')
        self.assertEqual(receipt['delivery_status'],'pending')
        receipt=record_delivery(self.root/'run-1',status='failed',message_id=None)
        self.assertEqual(receipt['artifact_status'],'verified')
        self.assertEqual(receipt['delivery_status'],'failed')
        self.assertIn('Author: Zeno Ren',(self.root/'run-1/report.md').read_text())
    def test_missing_or_corrupt_file_never_reports_saved(self):
        archive_report(self.root,'run-1',self.summary)
        p=self.root/'run-1/report.md';p.write_text('corrupt')
        self.assertEqual(verify_archive(p.parent)['artifact_status'],'hash_mismatch')
        p.unlink()
        self.assertEqual(verify_archive(p.parent)['artifact_status'],'missing')
        with self.assertRaises(ValueError):
            record_delivery(p.parent,status='delivered',message_id='synthetic-1')
    def test_immutable_retries_do_not_overwrite_history(self):
        first=archive_report(self.root,'run-1',self.summary)
        self.assertEqual(archive_report(self.root,'run-1',self.summary),first)
        with self.assertRaises(FileExistsError):
            archive_report(self.root,'run-1',{**self.summary,'policy_version':'different'})
        with self.assertRaises(ValueError):archive_report(self.root,'../outside',self.summary)
    def test_delivered_requires_receipt_and_cannot_be_overwritten(self):
        archive_report(self.root,'run-1',self.summary)
        with self.assertRaises(ValueError):record_delivery(self.root/'run-1',status='delivered',message_id=None)
        record_delivery(self.root/'run-1',status='delivered',message_id='synthetic-1')
        with self.assertRaises(ValueError):record_delivery(self.root/'run-1',status='delivered',message_id='synthetic-2')
    def test_incomplete_manifest_cannot_claim_verified(self):
        archive_report(self.root,'run-1',self.summary)
        p=self.root/'run-1/receipt.json';receipt=json.loads(p.read_text());receipt['artifacts']={}
        p.write_text(json.dumps(receipt))
        with self.assertRaises(ValueError):verify_archive(p.parent)
