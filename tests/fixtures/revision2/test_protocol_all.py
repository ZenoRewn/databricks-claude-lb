"""Red-first contracts. Synthetic real upstream and downstream sockets, baked main only.
Failure is expected on both production and resilience images; not a release suite.
"""
import asyncio,json,random,unittest
from protocol_socket_all import run_case,event

MODES=[('copilot','responses'),('azure','responses'),('copilot','chat'),('azure','chat'),('databricks','messages')]

def terminal(api,nl=b'\n'):
    if api=='responses': return event(nl=nl)
    if api=='chat': return b'data: [DONE]'+nl+nl
    return b'event: message_stop'+nl+b'data: {"type":"message_stop"}'+nl+nl

def delta(api):
    if api=='messages': return b'event: message_start\ndata: {"type":"message_start","message":{"id":"synthetic","usage":{"input_tokens":1}}}\n\n'
    if api=='chat': return b'data: {"choices":[{"delta":{"content":"synthetic"}}]}\n\n'
    return b'data: {"type":"response.output_text.delta","delta":"'+ '汉😀'.encode()+b'"}\n\n'

class AllProviderContracts(unittest.IsolatedAsyncioTestCase):
    async def check(self,label,wire_for,check,*,only=None,fragments='single',gzip_body=False,abrupt=False):
        for provider,api in MODES:
            if only and api not in only: continue
            with self.subTest(provider=provider,api=api):
                wire=wire_for(api)
                r=await asyncio.wait_for(run_case(label,wire,fragments,gzip_body,abrupt,provider,api),15)
                print('RESULT '+json.dumps(r,sort_keys=True),flush=True)
                # These assertions run BEFORE the defect-specific contract and client teardown
                # is excluded by the harness's recorded resource counts.
                self.assertEqual(r['pool_requests_before_teardown'],0)
                self.assertEqual(r['business_active_before_teardown'],0)
                self.assertEqual(r['registry_before_teardown'],0)
                self.assertEqual(r['settlement_calls'],1)
                self.assertEqual(r['source_requests'],1,'no replay after upstream execution')
                check(r)

    def success(self,r):
        self.assertTrue(r['exact_wire'],'complete upstream wire must pass through without appended false error')
        self.assertEqual(r['endpoint_errors'],0)
        self.assertEqual(r['endpoint_successes'],1)

    def incomplete(self,r):
        self.assertEqual(r['endpoint_successes'],0,'no terminal cannot count as success')
        self.assertGreaterEqual(r['downstream_error_events'],1,'new error must be independently dispatchable')
        self.assertEqual(r['downstream_completed_events'],0,'never complete an unfinished upstream terminal')

    async def test_lf_unicode_control(self):
        await self.check('lf_unicode',lambda a:delta(a)+terminal(a),self.success,fragments='byte')
    async def test_crlf_every_byte(self):
        await self.check('crlf',lambda a:delta(a)+terminal(a,b'\r\n'),self.success,fragments='byte')
    async def test_crlf_usage_all_providers(self):
        def wire(api):
            if api=='responses': return terminal(api,b'\r\n')
            if api=='chat': return b'data: {"choices":[],"usage":{"prompt_tokens":1,"completion_tokens":1}}\r\n\r\n'+terminal(api,b'\r\n')
            return b'event: message_start\r\ndata: {"type":"message_start","message":{"usage":{"input_tokens":1}}}\r\n\r\nevent: message_delta\r\ndata: {"type":"message_delta","usage":{"output_tokens":1}}\r\n\r\n'+terminal(api,b'\r\n')
        def check(r):
            self.assertEqual(r['recorded_input_tokens'],1,'CRLF framing must preserve usage observation')
            self.assertEqual(r['recorded_output_tokens'],1)
            self.success(r)
        await self.check('crlf_usage',wire,check,fragments=7)
    async def test_cr_only(self):
        await self.check('cr',lambda a:delta(a)+terminal(a,b'\r'),self.success)
    async def test_bom_single(self):
        await self.check('bom_single',lambda a:b'\xef\xbb\xbf'+terminal(a),self.success)
    async def test_bom_fragmented_control(self):
        await self.check('bom_byte',lambda a:b'\xef\xbb\xbf'+terminal(a),self.success,fragments='byte')
    async def test_comments_then_terminal_control(self):
        await self.check('comments',lambda a:b': heartbeat\n\n'+terminal(a),self.success,fragments=7)
    async def test_empty_eof(self):
        await self.check('empty',lambda a:b'',self.incomplete)
    async def test_comments_only_eof(self):
        await self.check('comments_only',lambda a:b': heartbeat\n\n',self.incomplete)
    async def test_partial_data_line(self):
        await self.check('partial',lambda a:delta(a)+b'data: {"type":"unfinished',self.incomplete,fragments=7)
    async def test_unterminated_valid_looking_terminal(self):
        await self.check('terminal_tail',lambda a:terminal(a)[:-1],self.incomplete,fragments=7)
    async def test_truncated_utf8_tail(self):
        await self.check('utf8_tail',lambda a:delta(a)+b'data: {"delta":"\xf0\x9f',self.incomplete,fragments='byte')
    async def test_header_only_is_not_terminal(self):
        await self.check('header_only',lambda a:b'event: response.completed\n\n',self.incomplete,only={'responses'})
    async def test_invalid_json_is_not_terminal(self):
        await self.check('invalid_json',lambda a:b'event: response.completed\ndata: not-json\n\n',self.incomplete,only={'responses'})
    async def test_missing_completed_id_is_not_terminal(self):
        await self.check('missing_id',lambda a:b'data: {"type":"response.completed","response":{}}\n\n',self.incomplete,only={'responses'})
    async def test_payload_type_precedes_generic_header(self):
        await self.check('payload_precedence',lambda a:b'event: message\n'+terminal(a),self.success,only={'responses'})
    async def test_payload_delta_overrides_completed_header(self):
        await self.check('conflicting_completed_header',lambda a:b'event: response.completed\n'+delta(a),self.incomplete,only={'responses'})
    async def test_failed_is_not_success(self):
        def check(r):
            self.assertTrue(r['exact_wire'],'preserve valid upstream failure without another local failure')
            self.assertEqual(r['endpoint_successes'],0,'failed terminal != success')
            self.assertEqual(r['endpoint_errors'],0,'explicit invalid_prompt is request-local')
        await self.check('failed',lambda a:event('response.failed'),check,only={'responses'},fragments=7)
    async def test_incomplete_is_not_success(self):
        def check(r):
            self.assertTrue(r['exact_wire'])
            self.assertEqual(r['endpoint_successes'],0)
            self.assertEqual(r['endpoint_errors'],0,'max_output_tokens is request-local')
        await self.check('incomplete',lambda a:event('response.incomplete'),check,only={'responses'},fragments=7)
    async def test_native_error_preserved(self):
        def check(r):
            self.assertTrue(r['exact_wire'],'valid upstream error is not absent-terminal truncation')
            self.assertEqual(r['endpoint_successes'],0)
        await self.check('native_error',lambda a:b'event: error\ndata: {"type":"error","code":"server_error","message":"synthetic","error":{"type":"api_error","message":"synthetic"}}\n\n',check)
    async def test_large_plain_control(self):
        await self.check('large_plain',lambda a:delta(a)+b'data: {"type":"synthetic.delta","delta":"'+b'x'*150000+b'"}\n\n'+terminal(a),self.success)
    async def test_large_gzip(self):
        await self.check('large_gzip',lambda a:delta(a)+b'data: {"type":"synthetic.delta","delta":"'+b'x'*150000+b'"}\n\n'+terminal(a),self.success,gzip_body=True)
    async def test_post_terminal_transport_error(self):
        await self.check('post_terminal_error',terminal,self.success,abrupt=True)
    async def test_pre_terminal_transport_error(self):
        await self.check('pre_terminal_error',lambda a:delta(a)+b'data: {"delta":"unfinished',self.incomplete,abrupt=True)

if __name__=='__main__':
    import signal; signal.alarm(200); unittest.main(verbosity=2)
