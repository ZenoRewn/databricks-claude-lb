"""Approved operational limits and protocol contracts; synthetic inputs only."""
import asyncio
from contextlib import aclosing, closing
import gzip
import json
import os
import random
import subprocess
import sys
import time
import unittest
from unittest.mock import AsyncMock, patch

import httpx
import main

MODES = [('copilot','responses'), ('azure','responses'), ('copilot','chat'), ('azure','chat'), ('databricks','messages')]


def terminal(api='responses'):
    if api == 'messages':
        return b'event: message_stop\ndata: {"type":"message_stop"}\n\n'
    if api == 'chat':
        return b'data: [DONE]\n\n'
    return b'data: {"type":"response.completed","response":{"id":"synthetic"}}\n\n'


class Chunks(httpx.AsyncByteStream):
    def __init__(self, parts, error=None, wait=None):
        self.parts, self.error, self.wait = parts, error, wait
        self.closed = False
    async def __aiter__(self):
        for part in self.parts:
            yield part
            await asyncio.sleep(0)
        if self.error:
            raise self.error
        if self.wait:
            await self.wait.wait()
    async def aclose(self):
        self.closed = True


class FramerTests(unittest.TestCase):
    def framed(self, parts, limit=1024*1024):
        budget = main._RetainedStreamBudget(8*limit)
        framer = main._SSEFramer(budget, limit)
        frames = []
        try:
            for part in parts:
                frames.extend(framer.feed(part))
            frames.extend(framer.eof())
            return b''.join(frames)
        finally:
            framer.close()
            self.assertEqual(budget.retained, 0)

    def test_all_single_cuts_and_random_partitions(self):
        wire = b'\xef\xbb\xbf: comment\r\ndata: '+json.dumps({'delta':'汉😀'},ensure_ascii=False).encode()+b'\r\n\r\n'+terminal()+b': last\r\r'
        for cut in range(len(wire)+1):
            self.assertEqual(self.framed([wire[:cut],wire[cut:]]), wire)
        self.assertEqual(self.framed([wire[i:i+1] for i in range(len(wire))]), wire)
        rng=random.Random(20260907)
        for _ in range(200):
            cuts=sorted(set([0,len(wire)]+[rng.randrange(len(wire)+1) for _ in range(20)]))
            self.assertEqual(self.framed([wire[a:b] for a,b in zip(cuts,cuts[1:])]), wire)

    def test_complete_frame_only_and_discard_pending_eof(self):
        for tail in (terminal()[:-1], b'data: {"x":"\xf0\x9f', b': unfinished'):
            self.assertEqual(self.framed([b': complete\n\n',tail]), b': complete\n\n')

    def test_below_at_above_actual_8mib_cap(self):
        cap = 8*1024*1024
        self.assertEqual(main.PER_PENDING_EVENT,cap)
        for size in (cap-1,cap,cap+1):
            # 4 KiB source deliveries; entire fixture is a single comment event.
            wire=b':'+b'x'*(size-3)+b'\n\n'
            parts=[wire[i:i+4096] for i in range(0,len(wire),4096)]
            if size<=cap:
                self.assertEqual(len(self.framed(parts,cap)), size)
            else:
                with self.assertRaises(main._LocalStreamLimit):
                    self.framed(parts,cap)

    def test_utf8_bytes_not_characters(self):
        wire=b'data: "'+('😀'*20).encode()+b'"\n\n'
        self.assertEqual(self.framed([wire],len(wire)),wire)
        with self.assertRaises(main._LocalStreamLimit):
            self.framed([wire],len(wire)-1)

    def test_stream_sum_unlimited_and_prompt_frame_release(self):
        budget=main._RetainedStreamBudget(64)
        framer=main._SSEFramer(budget,32)
        wire=b': '+b'x'*27+b'\n\n'
        for _ in range(10000):
            self.assertEqual(list(framer.feed(wire)),[wire])
            self.assertEqual(budget.retained,0)
        framer.close()
        self.assertLessEqual(budget.peak,32)

    def test_budget_allocation_rejection_is_atomic(self):
        budget=main._RetainedStreamBudget(256)
        budget.acquire(255)
        with self.assertRaises(main._LocalStreamLimit): budget.acquire(2)
        self.assertEqual(budget.retained,255)
        budget.acquire(1)
        self.assertEqual(budget.retained,256)
        budget.release(256)

    def test_terminal_nested_schema_and_payload_type(self):
        valid={'type':'response.completed','response':{'id':'s','usage':{'input_tokens':1,'output_tokens':2,'total_tokens':3,'input_tokens_details':{'cached_tokens':0},'output_tokens_details':{'reasoning_tokens':1}},'usage_metadata':{'amount':'1','metadata':[]},'end_turn':True}}
        bad=[]
        for key,val in [('id',1),('end_turn',1),('usage',[]),('usage_metadata',{'amount':1}),('status','incomplete')]:
            v=json.loads(json.dumps(valid));v['response'][key]=val;bad.append(v)
        for val in [True,1.1,'1',2**63,None]:
            v=json.loads(json.dumps(valid));v['response']['usage']['input_tokens']=val;bad.append(v)
        for field,value in [('input_tokens_details',{}),('input_tokens_details',{'cached_tokens':False}),('input_tokens_details',{'cached_tokens':0,'cache_write_tokens':None}),('output_tokens_details',{}),('output_tokens_details',{'reasoning_tokens':'1'})]:
            v=json.loads(json.dumps(valid));v['response']['usage'][field]=value;bad.append(v)
        for key in ['item_id','call_id','delta','text','summary_index','content_index']:
            v=json.loads(json.dumps(valid));v[key]=[];bad.append(v)
        for value in bad:
            o=main._SSEObservation('responses');o.observe(b'data: '+json.dumps(value).encode()+b'\n\n')
            self.assertIsNone(o.terminal,value)
        o=main._SSEObservation('responses');o.observe(b'event: message\ndata: '+json.dumps(valid).encode()+b'\n\n')
        self.assertEqual(o.terminal,'completed')
        self.assertEqual((o.input_tokens,o.output_tokens),(1,2))

    def test_header_only_nonobject_done_multiline_and_nonfinite(self):
        for wire in [b'event: response.completed\n\n',b'event: response.completed\ndata: []\n\n',b'data: [DONE]\n\n',b'data: {"type":"response.completed","response":{"id":"x","usage":NaN}}\n\n',b'data: [DONE]\ndata: extra\n\n']:
            o=main._SSEObservation('responses');o.observe(wire);self.assertIsNone(o.terminal)
        o=main._SSEObservation('responses');o.observe(b'data: {"type":"response.completed",\ndata: "response":{"id":"x"}}\n\n');self.assertEqual(o.terminal,'completed')

    def test_configuration_rejects_unlimited_invalid_and_inconsistent(self):
        for raw in ['0','-1','nan','inf','1.5','',str(sys.maxsize+1),' 8']:
            with patch.dict(os.environ,{'PER_PENDING_EVENT':raw}):
                with self.assertRaises(ValueError): main._positive_byte_limit('PER_PENDING_EVENT',8)
        env=dict(os.environ,PER_PENDING_EVENT='257',PER_PROCESS_TOTAL_RETAINED_STREAM_BUFFER='256')
        result=subprocess.run([sys.executable,'-c','import main'],env=env,capture_output=True,timeout=10)
        self.assertNotEqual(result.returncode,0)
        self.assertIn(b'must not exceed',result.stderr)


class StreamPolicyTests(unittest.IsolatedAsyncioTestCase):
    async def make(self,provider,api,parts,*,error=None,headers=None,wait=None):
        if provider=='copilot': ep=main.CopilotEndpoint('synthetic','');cls=main.CopilotProxy
        elif provider=='azure': ep=main.AzureOpenAIEndpoint('synthetic','http://synthetic','');cls=main.AzureOpenAIProxy
        else: ep=main.WorkspaceEndpoint('synthetic','http://synthetic','');cls=main.ClaudeProxy
        p=cls(main.LoadBalancer([ep],circuit_breaker_threshold=1),'')
        await p.client.aclose()
        stream=Chunks(parts,error,wait)
        calls=[]
        async def handler(req):
            calls.append(req)
            return httpx.Response(200,stream=stream,headers=headers)
        p.client=httpx.AsyncClient(transport=httpx.MockTransport(handler),trust_env=False)
        p._build_headers=AsyncMock(return_value={})
        p._probe_upstream_connect=AsyncMock(return_value={'ok':True})
        p.load_balancer.on_request_end=AsyncMock(wraps=p.load_balancer.on_request_end)
        self.addAsyncCleanup(p.client.aclose)
        return p,ep,stream,calls

    async def response(self,p,ep,provider,api):
        await p.load_balancer.on_request_start(ep)
        if provider=='databricks':
            return await p._stream_request(ep,'http://synthetic/messages',{}, {},model='synthetic',start_time=time.time())
        return await p._stream_response(ep,'http://synthetic/responses',{}, {},'synthetic',api,time.time())

    async def run_mode(self,provider,api,parts,**kw):
        p,ep,stream,calls=await self.make(provider,api,parts,**kw)
        response=await self.response(p,ep,provider,api)
        output=b''.join([b async for b in response.body_iterator])
        self.assertTrue(stream.closed)
        self.assertEqual(ep.active_requests,0)
        self.assertEqual(p.load_balancer.on_request_end.await_count,1)
        self.assertEqual(len(calls),1)
        self.assertEqual(getattr(p,'_stream_connections',{}),{})
        self.assertEqual(main._STREAM_BUFFER_BUDGET.retained,0)
        return output,p,ep

    async def test_lowered_event_caps_all_modes(self):
        for provider,api in MODES:
            for size in (255,256,257):
                with self.subTest(provider=provider,api=api,size=size):
                    wire=b':'+b'x'*(size-3)+b'\n\n'
                    with patch.object(main,'PER_PENDING_EVENT',256):
                        output,p,ep=await self.run_mode(provider,api,[wire,terminal(api)])
                    if size<=256:
                        self.assertEqual(output,wire+terminal(api));self.assertEqual(ep.successful_requests,1)
                    else:
                        self.assertIn(b'protocol_buffer_limit',output);self.assertNotIn(wire,output)
                        self.assertEqual(ep.neutral_requests,1);self.assertEqual(ep.successful_requests,0)
                    self.assertEqual(ep.total_errors,0)

    async def test_aggregate_chunk_pressure_all_modes(self):
        for provider,api in MODES:
            budget=main._RetainedStreamBudget(256)
            with self.subTest(provider=provider,api=api),patch.object(main,'_STREAM_BUFFER_BUDGET',budget):
                output,p,ep=await self.run_mode(provider,api,[b'x'*257])
                self.assertIn(b'protocol_buffer_limit',output)
                self.assertEqual(ep.neutral_requests,1);self.assertEqual(ep.total_errors,0)
                self.assertLessEqual(budget.peak,256)

    async def test_observer_bug_neutral_all_modes(self):
        for provider,api in MODES:
            with self.subTest(provider=provider,api=api),patch.object(main,'_parse_sse_event_block',side_effect=RuntimeError('synthetic')):
                output,p,ep=await self.run_mode(provider,api,[terminal(api)])
                self.assertIn(b'local_observer_error',output);self.assertEqual(ep.total_errors,0)
                self.assertEqual(ep.neutral_requests,1)

    async def test_partial_unforwarded_data_never_authorizes_post_replay(self):
        for provider,api in MODES:
            with self.subTest(provider=provider,api=api):
                output,p,ep=await self.run_mode(provider,api,[b'data: {"unfinished":'],error=httpx.ConnectError('synthetic after headers'))
                self.assertNotIn(b'unfinished',output);self.assertEqual(ep.total_errors,1)
                self.assertEqual(ep.successful_requests,0)

    async def test_conservative_unknown_eof_and_remote_errors(self):
        for provider,api in MODES:
            for wire in (b'',b'event: error\ndata: {"type":"error","code":"rate_limit_exceeded","message":"synthetic"}\n\n'):
                output,p,ep=await self.run_mode(provider,api,[wire])
                self.assertEqual(ep.total_errors,1);self.assertEqual(ep.successful_requests,0)

    async def test_half_open_overflow_one_trial_neutral_once(self):
        p,ep,stream,calls=await self.make('copilot','responses',[b'x'*257])
        p.load_balancer._open(ep);ep.circuit_retry_at=0
        responses=await asyncio.gather(*(p.proxy_responses({'model':'synthetic'},stream=True) for _ in range(100)),return_exceptions=True)
        accepted=[r for r in responses if not isinstance(r,Exception)]
        self.assertEqual(len(accepted),1)
        self.assertEqual(sum(isinstance(r,main.HTTPException) and r.status_code==503 for r in responses),99)
        with patch.object(main,'PER_PENDING_EVENT',256):
            body=b''.join([b async for b in accepted[0].body_iterator])
        self.assertIn(b'protocol_buffer_limit',body)
        self.assertEqual(ep.neutral_requests,1);self.assertEqual(ep.total_errors,0)
        self.assertEqual(ep.active_requests,0);self.assertFalse(ep.half_open_in_flight)
        self.assertEqual(p.load_balancer.on_request_end.await_count,1)
        self.assertEqual(main._STREAM_BUFFER_BUDGET.retained,0)
        self.assertEqual(len(calls),1)

    async def test_shared_budget_pressure_does_not_cancel_sibling(self):
        budget=main._RetainedStreamBudget(256)
        wait=asyncio.Event()
        p,ep,stream,calls=await self.make('copilot','responses',[b'x'*120],wait=wait)
        with patch.object(main,'_STREAM_BUFFER_BUDGET',budget),patch.object(main,'STREAM_HEARTBEAT_INTERVAL',.01):
            first=await self.response(p,ep,'copilot','responses')
            self.assertTrue((await anext(first.body_iterator)).startswith(b':'))
            self.assertEqual(budget.retained,120)
            # Only second stream exceeds aggregate; first remains live.
            p2,ep2,s2,c2=await self.make('azure','responses',[b'y'*200])
            second=await self.response(p2,ep2,'azure','responses')
            body=b''.join([b async for b in second.body_iterator])
            self.assertIn(b'protocol_buffer_limit',body);self.assertEqual(ep2.neutral_requests,1)
            self.assertFalse(stream.closed);self.assertEqual(ep.active_requests,1)
            self.assertEqual(budget.retained,120)
            await first.body_iterator.aclose()
            self.assertTrue(stream.closed);self.assertEqual(budget.retained,0)
            self.assertEqual(ep.cancelled_requests,1)

    async def test_gzip_bounded_decoding_multimember_and_large_remainder(self):
        wire=b':'+b'x'*(2*1024*1024)+b'\n\n'+terminal()
        for encoded in (gzip.compress(wire),gzip.compress(wire[:1000])+gzip.compress(wire[1000:])):
            response=httpx.Response(200,stream=Chunks([encoded]),headers={'content-encoding':'gzip'})
            decoded=[chunk async for chunk in main._decoded_stream_chunks(response)]
            self.assertEqual(b''.join(decoded),wire)
            self.assertLessEqual(max(map(len,decoded)),16384)
        for provider,api in MODES:
            wire=b':'+b'x'*150000+b'\n\n'+terminal(api)
            body,p,ep=await self.run_mode(provider,api,[gzip.compress(wire)],headers={'content-encoding':'gzip'})
            self.assertEqual(body,wire);self.assertEqual(ep.successful_requests,1)

    async def test_gzip_expansion_over_event_cap_neutral_and_closed(self):
        for provider,api in MODES:
            wire=b':'+b'x'*(1024*1024)+b'\n\n'+terminal(api)
            with patch.object(main,'PER_PENDING_EVENT',4096):
                body,p,ep=await self.run_mode(provider,api,[gzip.compress(wire)],headers={'content-encoding':'gzip'})
            self.assertIn(b'protocol_buffer_limit',body);self.assertEqual(ep.total_errors,0)
            self.assertEqual(ep.neutral_requests,1)

    async def test_complete_terminal_settled_before_client_stops_reading(self):
        for provider,api in MODES:
            p,ep,stream,calls=await self.make(provider,api,[terminal(api)],wait=asyncio.Event())
            response=await self.response(p,ep,provider,api)
            self.assertEqual(await anext(response.body_iterator),terminal(api))
            self.assertEqual(ep.successful_requests,1);self.assertEqual(ep.active_requests,0)
            await response.body_iterator.aclose()
            self.assertTrue(stream.closed);self.assertEqual(ep.cancelled_requests,0)
            self.assertEqual(main._STREAM_BUFFER_BUDGET.retained,0)

    async def test_leading_bom_after_local_heartbeat_remains_dispatchable(self):
        for provider,api in MODES:
            p,ep,stream,calls=await self.make(provider,api,[])
            release=asyncio.Event()
            class Delayed(Chunks):
                async def __aiter__(self):
                    await release.wait()
                    yield b'\xef\xbb\xbf'+terminal(api)
            delayed=Delayed([])
            async def handler(req):
                return httpx.Response(200,stream=delayed)
            await p.client.aclose()
            p.client=httpx.AsyncClient(transport=httpx.MockTransport(handler),trust_env=False)
            self.addAsyncCleanup(p.client.aclose)
            with patch.object(main,'STREAM_HEARTBEAT_INTERVAL',.01):
                response=await self.response(p,ep,provider,api)
                heartbeat=await anext(response.body_iterator)
                self.assertTrue(heartbeat.startswith(b':'))
                release.set()
                body=b''.join([b async for b in response.body_iterator])
            self.assertEqual(body,terminal(api))
            self.assertEqual(ep.successful_requests,1)
            self.assertTrue(delayed.closed)
            self.assertEqual(main._STREAM_BUFFER_BUDGET.retained,0)

    async def test_oversized_delivery_closes_upstream_before_consumer_resumes(self):
        budget=main._RetainedStreamBudget(256)
        allow=asyncio.Event()
        class DelayedLarge(Chunks):
            async def __aiter__(self):
                yield b': first\n\n'
                await allow.wait()
                yield b'x'*1024*1024
        stream=DelayedLarge([])
        upstream=httpx.Response(200,stream=stream)
        with patch.object(main,'_STREAM_BUFFER_BUDGET',budget):
            frames=main._framed_sse(upstream)
            self.assertEqual(await anext(frames),b': first\n\n')
            allow.set()
            async with asyncio.timeout(2):
                while not stream.closed: await asyncio.sleep(.001)
            self.assertLessEqual(budget.retained,256)
            with self.assertRaises(main._LocalStreamLimit): await anext(frames)
            await frames.aclose()
            self.assertEqual(budget.retained,0)

    async def test_random_chunks_all_modes(self):
        rng=random.Random(44)
        for provider,api in MODES:
            for _ in range(10):
                wire=b'\xef\xbb\xbf: '+('汉😀'*20).encode()+b'\r\n\r\n'+terminal(api)
                cuts=sorted(set([0,len(wire)]+[rng.randrange(len(wire)+1) for _ in range(30)]))
                body,p,ep=await self.run_mode(provider,api,[wire[a:b] for a,b in zip(cuts,cuts[1:])])
                self.assertEqual(body,wire);self.assertEqual(ep.successful_requests,1)


if __name__=='__main__': unittest.main()
