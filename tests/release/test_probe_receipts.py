"""Read failed acceptance receipts without repeating inference. Author: Zeno Ren."""
import copy
import json
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from operations.release.backend import Backend
from operations.release.engine import Engine,OwnershipLost,UnsafeState,new_record


class ProbeReceiptTests(unittest.TestCase):
    def backend(self):
        b=object.__new__(Backend);b.namespace='lab';b.plan={'container':'lb'}
        b.check_owner=Mock();pod={'metadata':{'name':'lb','uid':'original'}}
        b.api=SimpleNamespace(get=Mock(return_value=pod),exec=Mock())
        return b,pod

    def test_failed_exec_reads_receipt_without_a_second_business_probe(self):
        b,pod=self.backend();saved={'verified':False,'failed':True,'error_code':'public_protocol_acceptance_failed',
                                   'checks':[{'completed':True,'marker_found':False,'request_id':'r'}]}
        b.api.exec.side_effect=[RuntimeError('bounded_probe_failed'),json.dumps(saved)]
        result=b._probe(pod,'business',extra=('--plan','{}'))
        self.assertEqual(result,saved)
        self.assertEqual([call.args[3][2] for call in b.api.exec.call_args_list],['business','receipt'])

    def test_readback_failure_keeps_original_failure_and_never_replays(self):
        b,pod=self.backend();b.api.exec.side_effect=[RuntimeError('bounded_probe_failed'),OSError('read failed')]
        with self.assertRaisesRegex(RuntimeError,'bounded_probe_failed'):b._probe(pod,'business',extra=('--plan','{}'))
        self.assertEqual(b.api.exec.call_count,2)

    def test_lost_lease_or_replaced_pod_blocks_receipt_exec(self):
        for replaced in (False,True):
            with self.subTest(replaced=replaced):
                b,pod=self.backend();b.api.exec.side_effect=RuntimeError('bounded_probe_failed')
                if replaced:b.api.get.side_effect=[pod,{'metadata':{'name':'lb','uid':'other'}}]
                else:b.check_owner.side_effect=[None,OwnershipLost('lost')]
                with self.assertRaises(UnsafeState if replaced else OwnershipLost):b._probe(pod,'business',extra=('--plan','{}'))
                self.assertEqual(b.api.exec.call_count,1)

    def test_engine_persists_failed_checks_before_entering_recovery(self):
        saved=[];record=new_record({'release_id':'r','maintenance_seconds':300})
        record['phase']='verifying_business'
        receipt={'verified':False,'failed':True,'checks':[{'request_id':'r','completed':True,'marker_found':False}]}
        store=SimpleNamespace(read=lambda:copy.deepcopy(record),save=lambda r:saved.append(copy.deepcopy(r)))
        backend=SimpleNamespace(check_owner=lambda:None,observe=lambda r:{},perform=lambda a,r:receipt)
        result=Engine(store,backend).tick()
        self.assertEqual(result['phase'],'recovering')
        self.assertEqual(result['action_receipts']['verify_business'],receipt)
        self.assertTrue(any(r['phase']=='verifying_business' and r['action_receipts'].get('verify_business')==receipt for r in saved))
