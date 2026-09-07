"""Further independent encoded-wire and client parser edge oracles; baked app only."""
import asyncio,gzip,json,logging,unittest
from unittest.mock import patch
import main,protocol_socket_all as socket_probe
from test_protocol_all import MODES,terminal
logging.disable(logging.CRITICAL)
class Wire(unittest.IsolatedAsyncioTestCase):
 async def case(self,provider,api,wire,encoded=None):
  if encoded is None:r=await socket_probe.run_case('review_wire',wire,provider=provider,api_type=api)
  else:
   with patch.object(socket_probe.gzip,'compress',return_value=encoded):
    r=await socket_probe.run_case('review_encoded',wire,gzip_body=True,provider=provider,api_type=api)
  for k in ('pool_requests_before_teardown','business_active_before_teardown','registry_before_teardown'):self.assertEqual(r[k],0)
  self.assertEqual(r['source_requests'],1);self.assertEqual(r['settlement_calls'],1);self.assertEqual(main._STREAM_BUFFER_BUDGET.retained,0)
  return r
 async def test_gzip_invalid_header_bounded_error(self):
  for p,a in MODES:
   r=await self.case(p,a,b'data: {"pending":',b'not-gzip');self.assertEqual(r['endpoint_errors'],1);self.assertEqual(r['endpoint_successes'],0);self.assertGreaterEqual(r['downstream_error_events'],1)
  print('INVALID_GZIP_SOCKET_OK',5,flush=True)
 async def test_gzip_incomplete_member_bounded_error(self):
  wire=b'data: {"pending":"synthetic'
  for p,a in MODES:
   r=await self.case(p,a,wire,gzip.compress(wire)[:-8]);self.assertEqual(r['endpoint_errors'],1);self.assertEqual(r['endpoint_successes'],0);self.assertGreaterEqual(r['downstream_error_events'],1)
  print('INCOMPLETE_GZIP_SOCKET_OK',5,flush=True)
 async def test_multiple_gzip_members_are_preserved(self):
  for p,a in MODES:
   prefix=b': first\r\n\r\n';wire=prefix+terminal(a);r=await self.case(p,a,wire,gzip.compress(prefix)+gzip.compress(terminal(a)))
   self.assertTrue(r['exact_wire']);self.assertEqual(r['endpoint_successes'],1);self.assertEqual(r['endpoint_errors'],0)
  print('MULTIMEMBER_GZIP_SOCKET_OK',5,flush=True)
 async def test_real_gzip_exceeds_actual_8mib_event_cap_neutral(self):
  assert main.PER_PENDING_EVENT==8388608 and main._STREAM_BUFFER_BUDGET.limit==67108864
  for p,a in MODES:
   wire=b':'+b'x'*(9*1024*1024)+b'\n\n'+terminal(a);encoded=gzip.compress(wire)
   r=await self.case(p,a,wire,encoded);self.assertEqual(r['endpoint_successes'],0);self.assertEqual(r['endpoint_errors'],0);self.assertGreaterEqual(r['downstream_error_events'],1)
   self.assertFalse(r['added_truncation']);self.assertLess(r['downstream_bytes'],2048)
   print('ACTUAL_CAP_GZIP_SOCKET_OK',p,a,'raw',len(wire),'gzip',len(encoded),'charged_peak',main._STREAM_BUFFER_BUDGET.peak,flush=True)
 async def test_surrogate_and_duplicate_discriminator_not_client_valid(self):
  wires=[b'data: {"type":"response.completed","response":{"id":"\\ud800"}}\n\n',b'data: {"type":"response.failed","type":"response.completed","response":{"id":"s"}}\n\n']
  for p in ('copilot','azure'):
   for i,w in enumerate(wires):
    with self.subTest(provider=p,case=i):
     r=await self.case(p,'responses',w)
     print('JSON_CLIENT_MISMATCH',p,i,'successes',r['endpoint_successes'],flush=True)
     self.assertEqual(r['endpoint_successes'],0,'pinned serde event parsing rejects this payload')
if __name__=='__main__':
 import signal;signal.alarm(90);unittest.main(verbosity=2)
