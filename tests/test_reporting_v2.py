"""Typed monitoring never mistakes a draining gauge for a counter reset."""
import unittest


class TypedReportingTests(unittest.TestCase):
    def samples(self, name, values, kind='gauge', labels=None):
        return [{'timestamp':f'2026-09-20T00:0{i}:00Z','pod_uid':'pod-a','container_start_time':'2026-09-20T00:00:00Z',
                 'collection_status':'ok','metrics':[{'name':name,'type':kind,'unit':'events','labels':labels or {},'value':v}]}
                for i,v in enumerate(values)]

    def calculate(self,samples):
        from operations.reporting import calculate_window_v2
        return calculate_window_v2(samples,'2026-09-20T00:00:00Z','2026-09-20T00:02:00Z')

    def test_gauge_decrease_is_observation_not_reset(self):
        r=self.calculate(self.samples('lb_usage_pending_events',[8,4,1]))
        series=next(iter(r['targets']['pod-a']['series'].values()))
        self.assertEqual(series['latest'],1);self.assertEqual(series['minimum'],1);self.assertEqual(series['maximum'],8)
        self.assertNotIn('observed_delta',series);self.assertEqual(series['status'],'complete')

    def test_labeled_counters_do_not_collapse_outcomes(self):
        samples=self.samples('lb_requests_finished_total',[1,3,5],'counter',{'api_type':'responses','outcome':'completed'})
        for i,s in enumerate(samples):s['metrics'].append({'name':'lb_requests_finished_total','type':'counter','unit':'requests','labels':{'api_type':'responses','outcome':'failed'},'value':i})
        series=list(self.calculate(samples)['targets']['pod-a']['series'].values())
        self.assertEqual(sorted(x['observed_delta'] for x in series),[2,4])

    def test_restart_and_missing_collection_are_not_zero(self):
        samples=self.samples('lb_requests_started_total',[1,3,0],'counter')
        samples[2]['container_start_time']='2026-09-20T00:01:30Z'
        value=next(iter(self.calculate(samples)['targets']['pod-a']['series'].values()))
        self.assertEqual(value['observed_delta'],2);self.assertEqual(value['status'],'partial')
        samples[1]['collection_status']='failed'
        self.assertIsNone(next(iter(self.calculate(samples)['targets']['pod-a']['series'].values()))['observed_delta'])

    def test_replicas_remain_separate(self):
        samples=self.samples('lb_usage_pending_events',[8,4,1]);other=[{**s,'pod_uid':'pod-b'} for s in samples]
        r=self.calculate(samples+other);self.assertEqual(set(r['targets']),{'pod-a','pod-b'})
        self.assertNotIn('fleet_percentile',r)

    def test_parser_preserves_histogram_components_and_validates_labels(self):
        from operations.metrics_contract import parse_exposition
        values=parse_exposition('''# TYPE lb_request_duration_seconds histogram
lb_request_duration_seconds_bucket{api_type="messages",outcome="completed",le="1"} 2
lb_request_duration_seconds_sum{api_type="messages",outcome="completed"} 1.5
lb_request_duration_seconds_count{api_type="messages",outcome="completed"} 2
''')
        self.assertEqual(len(values),3);self.assertEqual(values[0]['labels']['le'],'1')
        with self.assertRaises(ValueError):parse_exposition('lb_requests_started_total{request_id="private"} 1')

    def test_conflicting_types_are_rejected(self):
        samples=self.samples('lb_usage_pending_events',[8,4,1]);samples[1]['metrics'][0]['type']='counter'
        with self.assertRaises(ValueError):self.calculate(samples)

    def test_histogram_components_share_coverage_and_units(self):
        from operations.metrics_contract import parse_exposition
        samples=self.samples('unused',[0,0,0])
        for i,s in enumerate(samples):
            s['metrics']=parse_exposition('\n'.join([
                f'lb_request_duration_seconds_bucket{{api_type="messages",outcome="completed",le="1"}} {i}',
                f'lb_request_duration_seconds_bucket{{api_type="messages",outcome="completed",le="+Inf"}} {i+1}',
                f'lb_request_duration_seconds_count{{api_type="messages",outcome="completed"}} {i+1}',
                f'lb_request_duration_seconds_sum{{api_type="messages",outcome="completed"}} {i+2}']))
        series=list(self.calculate(samples)['targets']['pod-a']['series'].values())
        self.assertTrue(all(s['observed_delta']==2 for s in series))
        self.assertEqual(next(s for s in series if s['component']=='count')['unit'],'observations')
        samples[1]['metrics']=samples[1]['metrics'][1:]
        series=list(self.calculate(samples)['targets']['pod-a']['series'].values())
        self.assertTrue(all(s['observed_delta'] is None for s in series))
