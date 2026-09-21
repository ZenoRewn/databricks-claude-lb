"""Kind-only synthetic model provider. Never routes outside the lab. Author: Zeno Ren."""
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
import threading
import time

settings={'delay':0,'status':200,'requests':0};lock=threading.Lock()


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        with lock:body=json.dumps(settings).encode()
        self.send_response(200);self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
    def do_POST(self):
        length=int(self.headers.get('Content-Length','0'))
        if not 0<=length<=65536:self.send_error(413);return
        body=json.loads(self.rfile.read(length))
        if self.path=='/control':
            with lock:
                settings.update({k:v for k,v in body.items() if k in ('delay','status')})
            self.do_GET();return
        with lock:
            settings['requests']+=1;delay=settings['delay'];status=settings['status']
        if delay:time.sleep(min(float(delay),120))
        api='messages' if '/messages' in self.path else 'responses' if '/responses' in self.path else 'chat'
        if status!=200:payload={'error':{'code':'synthetic_failure','message':'synthetic failure'}}
        elif api=='messages':payload={'id':'m-synthetic','type':'message','role':'assistant','model':body.get('model'),
            'content':[{'type':'text','text':'LB_OK'}],'stop_reason':'end_turn','usage':{'input_tokens':2,'output_tokens':3}}
        elif api=='responses':payload={'id':'r-synthetic','object':'response','status':'completed',
            'output':[{'type':'message','role':'assistant','content':[{'type':'output_text','text':'LB_OK'}]}],
            'usage':{'input_tokens':2,'output_tokens':3,'total_tokens':5}}
        else:payload={'id':'c-synthetic','object':'chat.completion','choices':[{'index':0,'message':{'role':'assistant','content':'LB_OK'},'finish_reason':'stop'}],
                      'usage':{'prompt_tokens':2,'completion_tokens':3,'total_tokens':5}}
        if body.get('stream') and status==200:
            if api=='messages':events=[('message_start',{'type':'message_start','message':{**payload,'content':[],'stop_reason':None}}),
                ('content_block_delta',{'type':'content_block_delta','index':0,'delta':{'type':'text_delta','text':'LB'}}),
                ('content_block_delta',{'type':'content_block_delta','index':0,'delta':{'type':'text_delta','text':'_OK'}}),
                ('message_delta',{'type':'message_delta','delta':{'stop_reason':'end_turn'},'usage':{'output_tokens':3}}),('message_stop',{'type':'message_stop'})]
            # Deliberately omit a full terminal text snapshot: the acceptance
            # check must reconstruct deltas instead of searching raw SSE bytes.
            elif api=='responses':events=[(None,{'type':'response.output_text.delta','delta':part}) for part in ('LB','_OK')]+[(None,{'type':'response.completed','response':{**payload,'output':[]}})]
            else:events=[(None,{'choices':[{'index':0,'delta':{'content':'LB'},'finish_reason':None}]}),
                         (None,{'choices':[{'index':0,'delta':{'content':'_OK'},'finish_reason':'stop'}],'usage':payload['usage']})]
            raw=''.join((f'event: {event}\n' if event else '')+'data: '+json.dumps(value)+'\n\n' for event,value in events).encode()
            if api=='chat':raw+=b'data: [DONE]\n\n'
            content_type='text/event-stream'
        else:raw=json.dumps(payload).encode();content_type='application/json'
        self.send_response(status);self.send_header('Content-Type',content_type);self.send_header('Content-Length',str(len(raw)));self.end_headers()
        try:self.wfile.write(raw)
        except (BrokenPipeError,ConnectionResetError):pass
    def log_message(self,*args):pass


if __name__=='__main__':ThreadingHTTPServer(('0.0.0.0',8000),Handler).serve_forever()
