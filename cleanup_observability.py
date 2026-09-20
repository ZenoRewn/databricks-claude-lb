"""Low-cardinality observations of cleanup owners. Author: Zeno Ren."""
import asyncio
from collections import Counter
import time
import weakref


class CleanupTracker:
    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self.active = {}
        self.seen = weakref.WeakSet()
        self.results = Counter()
        self.count = 0
        self.seconds = 0.0

    def watch(self, task):
        if task in self.seen:
            return
        self.seen.add(task)
        self.active[task] = self.clock()
        def finished(owner):
            self.seconds += max(0,self.clock()-self.active.pop(owner))
            self.count += 1
            self.results['cancelled' if owner.cancelled() else 'failed' if owner.exception() is not None else 'completed'] += 1
        task.add_done_callback(finished)

    def snapshot(self):
        return {'active':len(self.active), 'oldest_age_seconds':max(0,self.clock()-min(self.active.values())) if self.active else 0}

    def render(self):
        state=self.snapshot()
        lines=[]
        for key,value in state.items():
            name='lb_cleanup_'+key
            lines.extend([f'# HELP {name} Owned cleanup tasks; not connections',f'# TYPE {name} gauge',f'{name} {value}'])
        lines.extend(['# HELP lb_cleanup_finished_total Settled cleanup owners', '# TYPE lb_cleanup_finished_total counter'])
        lines.extend(f'lb_cleanup_finished_total{{result="{result}"}} {self.results[result]}' for result in ('completed','failed','cancelled'))
        lines.extend(['# HELP lb_cleanup_duration_seconds Cleanup owner lifetime', '# TYPE lb_cleanup_duration_seconds summary',
                      f'lb_cleanup_duration_seconds_count {self.count}',f'lb_cleanup_duration_seconds_sum {self.seconds}'])
        return '\n'.join(lines)+'\n'


CLEANUP = CleanupTracker()
