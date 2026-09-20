"""Reviewable changes, bounded probes, and private immutable planning files."""
import copy
import importlib
import json
from pathlib import Path
import stat
import tempfile
import unittest

from operations.release.model import APP_FILES,default_target_settings,target_pod_spec,validate_plan,SCHEMA


def plan():
    return {'schema_version':SCHEMA,'release_id':'r','namespace':'lb-lab','deployment':'lb','container':'lb',
            'services':['lb'],'source_revision':'a'*40,'image':'registry.invalid/lb@sha256:'+'b'*64,
            'source_hashes':{name:'c'*64 for name in APP_FILES},'snapshot_sha256':'d'*64,
            'target_settings':default_target_settings(),'direct_pod_access':False,'legacy_bootstrap':False,
            'storage_compatibility':'additive-compatible','public_urls':['https://example.invalid'],
            'business_probes':[{'api':'messages','model':'synthetic','stream':False,'max_tokens':32}]}


class PlanTests(unittest.TestCase):
    def test_default_budget_reserves_recovery(self):
        p=validate_plan(plan());self.assertEqual((p['maintenance_seconds'],p['forward_seconds'],p['drain_seconds']),(300,150,60))
    def test_unreviewed_commands_and_inadequate_budgets_are_rejected(self):
        values=[]
        p=plan();p['target_settings']['lifecycle']['preStop']['exec']['command']=['sh','-c','arbitrary'];values.append(p)
        p=plan();p['maintenance_seconds']=299;p['forward_seconds']=150;values.append(p)
        p=plan();p['direct_pod_access']=True;values.append(p)
        p=plan();p['image']='registry.invalid/lb:latest';values.append(p)
        p=plan();p['public_urls']=['https://user:password@example.invalid'];values.append(p)
        p=plan();p['public_urls']=['http://example.invalid'];values.append(p)
        p=plan();p['lab']=True;p['namespace']='production';values.append(p)
        p=plan();p['business_probes'][0]['max_tokens']=10000;values.append(p)
        for p in values:
            with self.subTest(p=p),self.assertRaises(ValueError):validate_plan(p)
    def test_generated_pod_spec_preserves_configuration_resources_and_input(self):
        current={'spec':{'template':{'spec':{'containers':[{'name':'lb','image':'old','env':[{'name':'PRIVATE','value':'fixture'}],
            'resources':{'requests':{'cpu':'100m','memory':'256Mi'}},'volumeMounts':[{'name':'cfg','mountPath':'/app/config.yaml'}]}],
            'volumes':[{'name':'cfg','secret':{'secretName':'fixture'}}],'terminationGracePeriodSeconds':30}}}}
        before=copy.deepcopy(current);target=target_pod_spec(current,plan())
        self.assertEqual(current,before)
        for name in ('env','resources','volumeMounts'):self.assertEqual(target['containers'][0][name],before['spec']['template']['spec']['containers'][0][name])
        self.assertEqual(target['volumes'],before['spec']['template']['spec']['volumes'])
        self.assertEqual(target['terminationGracePeriodSeconds'],90)
    def test_private_backup_cannot_overwrite_history(self):
        write=importlib.import_module('operations.release.__main__').private_write
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'snapshot.json';write(p,b'original')
            self.assertEqual(stat.S_IMODE(p.stat().st_mode),0o600)
            with self.assertRaises(FileExistsError):write(p,b'replacement')
            self.assertEqual(p.read_bytes(),b'original')
