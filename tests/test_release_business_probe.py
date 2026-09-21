"""Business acceptance inspects reconstructed text, never raw SSE substrings."""
import hashlib
import contextlib
import io
import json
from pathlib import Path
import tempfile
import os
import time
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch
import uuid

import httpx
import release_probe


def event(payload, newline='\n'):
    return ('data: '+json.dumps(payload)+newline+newline).encode()


def stream(api, parts=('LB','_OK'), *, terminal=True):
    if api=='messages':
        wire=b''.join(event({'type':'content_block_delta','index':0,'delta':{'type':'text_delta','text':part}}) for part in parts)
        end=event({'type':'message_delta','delta':{'stop_reason':'end_turn'}})+event({'type':'message_stop'})
    elif api=='responses':
        wire=b''.join(event({'type':'response.output_text.delta','output_index':0,'content_index':0,'delta':part}) for part in parts)
        end=event({'type':'response.completed','response':{'id':'r','status':'completed','output':[]}})
    else:
        wire=b''.join(event({'choices':[{'index':0,'delta':{'content':part},'finish_reason':None}]}) for part in parts)
        end=event({'choices':[{'index':0,'delta':{},'finish_reason':'stop'}]})+b'data: [DONE]\n\n'
    return wire+(end if terminal else b'')


class ByteStream(httpx.AsyncByteStream):
    def __init__(self,body):self.body=body
    async def __aiter__(self):
        for byte in self.body:yield bytes([byte])


class Cursor:
    def __init__(self,events):self.events=events
    async def __aenter__(self):return self
    async def __aexit__(self,*args):pass
    async def execute(self,*args):pass
    async def fetchall(self):
        payload=json.dumps({'events':self.events},ensure_ascii=False,sort_keys=True,separators=(',',':'))
        return [('batch',hashlib.sha256(payload.encode()).hexdigest(),payload)]


class BusinessProbeTests(unittest.IsolatedAsyncioTestCase):
    async def drive(self,api,wire,*,streaming=True,repeat=False):
        with tempfile.TemporaryDirectory() as directory:
            events=[];sends=[]
            async def upstream(request):
                sends.append(request)
                rid='probe-'+str(len(sends))
                events.append({'request_id':rid,'generation_outcome':'completed'})
                return httpx.Response(200,headers={'x-lb-request-id':rid,'content-type':'text/event-stream' if streaming else 'application/json'},stream=ByteStream(wire))
            client=httpx.AsyncClient(transport=httpx.MockTransport(upstream))
            def private_path(*parts):
                raw=Path(*parts)
                return Path(directory)/raw.name if str(raw).startswith('/tmp/lb-release-acceptance-') else raw
            cfg={'auth':{'api_key':'synthetic'},'usage_storage':{'type':'mysql'}}
            conn=SimpleNamespace(cursor=lambda:Cursor(events),close=lambda:None)
            plan={'release_id':'unit-'+uuid.uuid4().hex[:8],'public_urls':['https://probe.invalid'],
                  'business_probes':[{'api':api,'model':'synthetic','stream':streaming,'max_tokens':32}]}
            with patch.object(release_probe,'configuration',return_value=cfg),patch.object(release_probe,'Path',side_effect=private_path),\
                 patch.object(release_probe.httpx,'AsyncClient',return_value=client),patch.object(release_probe,'mysql_connection',AsyncMock(return_value=conn)):
                result=await release_probe.business_check(plan)
                if repeat:self.assertEqual(await release_probe.business_check(plan),result)
            self.assertEqual(len(sends),1)
            return result

    async def test_split_marker_is_accepted_for_every_api_without_replaying(self):
        for api in ('messages','responses','chat'):
            with self.subTest(api=api):
                wire=stream(api)
                self.assertNotIn(b'LB_OK',wire)
                result=await self.drive(api,wire,repeat=True)
                self.assertTrue(result['verified'])
                self.assertEqual(result['usage_events'],1)
                self.assertEqual(result['checks'][0]['text_chars'],5)

    async def test_metadata_is_not_the_generated_answer(self):
        for api in ('messages','responses','chat'):
            with self.subTest(api=api):
                wire=event({'metadata':{'probe':'LB_OK'}})+stream(api,('wrong',))
                with self.assertRaisesRegex(ValueError,'public_protocol_acceptance_failed'):
                    await self.drive(api,wire)

    async def test_visible_marker_still_requires_a_successful_terminal(self):
        for api in ('messages','responses','chat'):
            with self.subTest(api=api),self.assertRaisesRegex(ValueError,'public_protocol_acceptance_failed'):
                await self.drive(api,stream(api,('LB_OK',),terminal=False))

    async def test_buffered_metadata_cannot_satisfy_probe(self):
        payload={'id':'LB_OK','status':'completed','output':[{'type':'message','role':'assistant','content':[{'type':'output_text','text':'wrong'}]}]}
        with self.assertRaisesRegex(ValueError,'public_protocol_acceptance_failed'):
            await self.drive('responses',json.dumps(payload).encode(),streaming=False)


class ProbeTextTests(unittest.TestCase):
    def test_line_endings_bom_comments_and_json_escapes(self):
        for api in ('messages','responses','chat'):
            for newline in (b'\n',b'\r\n',b'\r'):
                with self.subTest(api=api,newline=newline):
                    raw=b'\xef\xbb\xbf: heartbeat\n\n'+stream(api,('LB_OK',)).replace(b'LB_OK',b'\\u004c\\u0042\\u005f\\u004f\\u004b')
                    result=release_probe.assess_probe_response(raw.replace(b'\n',newline),api,True)
                    self.assertTrue(result['completed']);self.assertTrue(result['marker_found'])
                    self.assertEqual(result['text_chars'],5)

    def test_snapshot_replaces_deltas_instead_of_duplicate_text(self):
        raw=stream('responses',terminal=False)+event({'type':'response.output_text.done','text':'LB_OK'})
        payload={'id':'r','status':'completed','output':[{'type':'message','role':'assistant','content':[{'type':'output_text','text':'LB_OK'}]}]}
        raw+=event({'type':'response.completed','response':payload})
        result=release_probe.assess_probe_response(raw,'responses',True)
        self.assertTrue(result['marker_found']);self.assertEqual(result['text_chars'],5)

    def test_alternative_choice_and_tool_arguments_are_not_primary_text(self):
        raw=event({'choices':[{'index':0,'delta':{'tool_calls':[{'function':{'arguments':'LB_OK'}}]}},
                               {'index':1,'delta':{'content':'LB_OK'}}]})+b'data: [DONE]\n\n'
        self.assertFalse(release_probe.assess_probe_response(raw,'chat',True)['marker_found'])

    def test_success_marker_does_not_override_error_incomplete_or_undispatched_terminal(self):
        for api in ('messages','responses','chat'):
            for raw in (stream(api).rstrip(b'\n'),stream(api)+event({'type':'error','error':{'message':'failed'}})):
                with self.subTest(api=api,raw=raw):
                    self.assertFalse(release_probe.assess_probe_response(raw,api,True)['completed'])
        raw=stream('chat',terminal=False)+event({'choices':[{'index':0,'delta':{},'finish_reason':'length'}]})+b'data: [DONE]\n\n'
        self.assertFalse(release_probe.assess_probe_response(raw,'chat',True)['completed'])

    def test_buffered_semantic_text_for_all_protocols(self):
        payloads={'messages':{'type':'message','stop_reason':'end_turn','content':[{'type':'text','text':'LB_OK'}]},
                  'responses':{'id':'r','status':'completed','output':[{'type':'message','role':'assistant','content':[{'type':'output_text','text':'LB_OK'}]}]},
                  'chat':{'choices':[{'index':0,'finish_reason':'stop','message':{'content':'LB_OK'}}]}}
        for api,payload in payloads.items():
            with self.subTest(api=api):
                result=release_probe.assess_probe_response(json.dumps(payload).encode(),api,False)
                self.assertTrue(result['completed']);self.assertTrue(result['marker_found'])


class ReceiptCLITests(unittest.TestCase):
    def test_failed_business_receipt_remains_nonzero_but_can_be_read_without_inference(self):
        plan={'release_id':'receipt-fixture'}
        with tempfile.TemporaryDirectory() as directory:
            cache=Path(directory)/'receipt.json'
            saved={'plan_hash':hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest(),
                   'verified':False,'execution_pid':os.getpid(),'checks':[{'completed':True,'marker_found':False,'request_id':'r'}]}
            cache.write_text(json.dumps(saved))
            with patch.object(release_probe,'_cache_path',return_value=cache),\
                 patch('sys.argv',['release_probe.py','business','--plan',json.dumps(plan)]),\
                 patch.object(release_probe,'run',AsyncMock(side_effect=ValueError('public_protocol_acceptance_failed'))),\
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(release_probe.main(),1)
            with patch.object(release_probe,'_cache_path',return_value=cache),\
                 patch('sys.argv',['release_probe.py','receipt','--plan',json.dumps(plan)]),\
                 patch.object(release_probe,'configuration',side_effect=AssertionError('No config/network read for receipt')),\
                 contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(release_probe.main(),0)
                result=json.loads(output.getvalue())
            self.assertTrue(result['failed']);self.assertFalse(result['verified'])
            self.assertEqual(result['error_code'],'public_protocol_acceptance_failed')
            self.assertEqual(result['checks'],saved['checks'])
            self.assertEqual(cache.stat().st_mode&0o777,0o600)
            with patch.object(release_probe,'_cache_path',return_value=cache),\
                 patch('sys.argv',['release_probe.py','business','--plan',json.dumps(plan)]),\
                 patch.object(release_probe,'configuration',return_value={'auth':{'api_key':'synthetic'}}),\
                 patch.object(release_probe.httpx,'AsyncClient',side_effect=AssertionError('Do not replay inference')),\
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(release_probe.main(),1)

    def test_business_pending_is_reported_as_pending_without_claiming_verified(self):
        with patch('sys.argv',['release_probe.py','business','--plan','{}']),\
             patch.object(release_probe,'run',AsyncMock(return_value={'verified':False,'pending':True})),\
             contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(release_probe.main(),0)
            result=json.loads(output.getvalue())
        self.assertFalse(result['verified']);self.assertTrue(result['pending'])

    def test_receipt_read_is_bound_to_the_original_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            cache=Path(directory)/'receipt.json';cache.write_text(json.dumps({'plan_hash':'different','verified':True}))
            with patch.object(release_probe,'_cache_path',return_value=cache),self.assertRaisesRegex(ValueError,'probe_plan_changed'):
                release_probe.read_receipt({'release_id':'r'})

    def test_receipt_read_preserves_running_or_unknown_execution_state(self):
        plan={'release_id':'r'}
        with tempfile.TemporaryDirectory() as directory:
            cache=Path(directory)/'receipt.json'
            cache.write_text(json.dumps({'plan_hash':hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest(),
                                        'verified':False,'execution_pid':123,'started_at':time.time()}))
            proc=SimpleNamespace(read_bytes=lambda:b'python\0/app/release_probe.py\0business\0{"release_id":"r"}')
            with patch.object(release_probe,'_cache_path',return_value=cache),patch.object(release_probe,'Path',return_value=proc):
                self.assertTrue(release_probe.read_receipt(plan)['pending'])
            with patch.object(release_probe,'_cache_path',return_value=cache),patch.object(release_probe,'Path',side_effect=OSError('gone')):
                self.assertTrue(release_probe.read_receipt(plan)['execution_uncertain'])
