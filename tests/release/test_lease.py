"""A fresh execution must fence even the same coordinator process identity."""
import copy
import unittest

from operations.release.kube import Lease
from operations.release.engine import OwnershipLost


class API:
    def __init__(self):self.obj=None
    def optional(self,*args):return copy.deepcopy(self.obj)
    def get(self,*args):return copy.deepcopy(self.obj)
    def create(self,kind,namespace,obj):
        self.obj=copy.deepcopy(obj);self.obj['metadata']['resourceVersion']='1'
    def replace(self,kind,namespace,obj):
        if obj['metadata']['resourceVersion']!=self.obj['metadata']['resourceVersion']:raise RuntimeError('conflict')
        self.obj=copy.deepcopy(obj);self.obj['metadata']['resourceVersion']=str(int(obj['metadata']['resourceVersion'])+1)


class LeaseTests(unittest.TestCase):
    def test_reacquire_same_identity_fences_previous_execution(self):
        api=API();old=Lease(api,'ops','workload','same-process');self.assertTrue(old.acquire())
        new=Lease(api,'ops','workload','same-process');self.assertTrue(new.acquire())
        self.assertGreater(new.epoch,old.epoch)
        with self.assertRaises(OwnershipLost):old.check()
    def test_other_live_owner_cannot_be_preempted(self):
        api=API();first=Lease(api,'ops','workload','first');self.assertTrue(first.acquire())
        self.assertFalse(Lease(api,'ops','workload','other').acquire())
