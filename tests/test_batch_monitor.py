import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import subprocess
import sys

from autoresearch_budget.batch_monitor import snapshot, summarize


class BatchMonitorTest(unittest.TestCase):
    def events(self):
        data = [('model_response', {'model_role': 'actor', 'raw_body': json.dumps({'provider': 'DeepInfra', 'service_tier': 'default'})}),
                ('advice_requested', {'actor_call': 10}),
                ('model_response', {'model_role': 'advisor', 'raw_body': json.dumps({'provider': 'OpenAI', 'service_tier': 'flex'})}),
                ('advice_decision', {'actor_call': 11, 'advice_id': 'advice-1', 'decision': 'adopt'}),
                ('patch_snapshot', {})]
        return '\n'.join(json.dumps({'kind': kind, 'payload': payload, 'sequence': i, 'timestamp': 'recent'})
                         for i, (kind, payload) in enumerate(data, 1))+'\n'

    def test_route_decision_and_partial_event_reporting(self):
        result = summarize({}, self.events().splitlines())
        self.assertEqual(result['checkpoints'], [10])
        self.assertEqual(result['decisions'][0]['decision'], 'adopt')
        self.assertEqual(result['routes'][1]['service_tier'], 'flex')
        self.assertTrue(result['event_sequence_contiguous'])
        self.assertFalse(summarize({}, [*self.events().splitlines(), '{'])['event_sequence_contiguous'])

    def test_requested_advice_reports_consumption_and_denials_without_checkpoints(self):
        entries=[('actor_advice_request',{'actor_call':2,'question':'Why?','evidence_ids':['observation-1']}),
                 ('advice_requested',{'actor_call':2,'reason':'advice_requested'}),
                 ('advice_delivered',{}),('actor_advice_request',{'actor_call':4}),
                 ('advice_denied',{'reason':'call_limit'})]
        events=[json.dumps({'sequence':i,'kind':kind,'payload':payload}) for i,(kind,payload) in enumerate(entries,1)]
        result=summarize({'max_advice_calls':1},events)
        self.assertEqual(result['checkpoints'],[])
        self.assertEqual(len(result['advice_requests']),2)
        self.assertEqual(result['advice_dispatched'],1)
        self.assertEqual(result['advice_delivered'],1)
        self.assertEqual(result['advice_calls_remaining'],0)
        self.assertEqual(result['advice_denials'],{'call_limit':1})

    def test_all_four_live_tasks_are_inspected_with_local_fallback(self):
        with TemporaryDirectory() as temp:
            root = Path(temp); active = []
            for i in range(4):
                job = root/f'job-{i}'
                p = job/f'trial-{i}/agent/advice';p.mkdir(parents=True)
                (p/'summary.json').write_text(json.dumps({'actor': {'api_calls': i}}))
                (p/'events.jsonl').write_text(self.events())
                active.append({'task_id': str(i), 'job_name': job.name, 'job_dir': str(job)})
            (root/'status.json').write_text(json.dumps({'state': 'running', 'active_tasks': active}))
            calls = []; real_run = subprocess.run
            def docker(command, **kwargs):
                calls.append(command)
                if command[1] == 'ps':
                    return subprocess.CompletedProcess(command, 0, '\n'.join(json.dumps({'ID': str(i), 'Names': f'trial-{i}__env-main-1'}) for i in range(4)))
                if command[2] == '2':
                    raise subprocess.CalledProcessError(1, command)
                # Compile the exact embedded inspector and exercise it against local fixture files.
                program = command[-1].replace('/logs/agent/advice', str(root/f'job-{command[2]}/trial-{command[2]}/agent/advice'))
                return real_run([sys.executable, '-c', program], **kwargs)
            with patch('subprocess.run', side_effect=docker):
                result = snapshot(root, live=True)
            self.assertEqual(len(result['progress']), 4)
            self.assertEqual(len(calls), 5)
            self.assertEqual(result['progress'][0]['source'], 'live_container')
            self.assertEqual(result['progress'][2]['source'], 'host_records')
            self.assertEqual(result['progress'][2]['live_error'], 'CalledProcessError')
            self.assertEqual(result['progress'][3]['checkpoints'], [10])
            with patch('subprocess.run', side_effect=AssertionError('no Docker without --live')):
                self.assertEqual(len(snapshot(root)['progress']), 4)


if __name__ == '__main__':unittest.main()
