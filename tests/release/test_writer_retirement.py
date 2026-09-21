"""An unscheduled pod needs deletion observation, never a guessed node name."""
import copy
from types import SimpleNamespace
import unittest

from operations.release.backend import Backend
from operations.release.engine import PendingOperation


class WriterRetirementTests(unittest.TestCase):
    def backend(self,deleting=False):
        pod={'metadata':{'name':'pending','uid':'pod-uid','finalizers':['lb.zeno.ink/observe-termination']},
             'spec':{'containers':[{'name':'lb'}]},'status':{'phase':'Pending'}}
        if deleting:pod['metadata']['deletionTimestamp']='2026-09-20T00:00:00Z'
        b=object.__new__(Backend);b.namespace='lb-lab';b.deployment='lb';b.plan={'container':'lb'}
        b.store=SimpleNamespace(save=lambda record:None)
        b._claim=lambda kind,obj:obj
        b._owned=lambda kind,name:({'spec':{'replicas':0}} if kind=='deployment' else copy.deepcopy(pod))
        writes=[];b._patch=lambda kind,obj,ops:writes.append(ops)
        def get(kind,namespace,name):
            if kind!='pod':raise AssertionError('No Node lookup before scheduling')
            return copy.deepcopy(pod)
        b.api=SimpleNamespace(optional=get,get=get)
        return b,pod,writes
    def test_scale_ack_before_delete_timestamp_remains_pending(self):
        b,pod,writes=self.backend();record={'action_receipts':{}}
        with self.assertRaises(PendingOperation):b._stop_writer(record,pod,'new')
        self.assertFalse(record['action_receipts']['writer_termination'])
        self.assertEqual(writes,[])
    def test_deleting_never_scheduled_pod_gets_explicit_proof(self):
        b,pod,writes=self.backend(True);record={'action_receipts':{}}
        b._stop_writer(record,pod,'new')
        self.assertTrue(record['action_receipts']['writer_termination']['pod-uid']['never_scheduled_after_deletion'])
        self.assertEqual(writes[0][0]['value'],[])
