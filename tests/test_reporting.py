import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from autoresearch_budget.reporting import report, compare, render
from autoresearch_budget.cli import main


class ReportingTest(unittest.TestCase):
    def fixture(self,root,task='owner__repo-1',reward=1,*,cost='0.25',unknown=False,provenance=True,condition='hybrid'):
        job=root/'jobs'/task; trial=job/'trial'; agent=trial/'agent';agent.mkdir(parents=True)
        records=root/'jobs/_poc_records'/task;records.mkdir(parents=True)
        (agent/'advice').mkdir();(agent/'advice/events.jsonl').write_text('')
        metadata={'task_id':task,'profile':'small','attempt':1,'budget_enforced':False,
                  'configuration':{'agent':{'max_steps':80},'consultation':{'policy':condition}}}
        if provenance:metadata['provenance']={'task_package_sha256':'same-task','runtime':{'source_sha256':{'a.py':'hash'}}}
        (records/'metadata.json').write_text(json.dumps(metadata))
        metrics={'stop_reason':'Submitted','actor':{'api_calls':2,'responses_with_cost':1 if unknown else 2,'cost_usd':None if unknown else cost,
                  'observed_cost_usd':cost},'advisor':{'logical_calls':1,'physical_api_calls':2,'unknown_cost_calls':1 if unknown else 0},
                  'total_cost_usd':None if unknown else cost}
        (records/'summary.json').write_text(json.dumps({'status':'completed','return_code':0,'metrics':metrics,'duration_seconds':3}))
        (agent/'mini-swe-agent.trajectory.json').write_text(json.dumps({'info':{'poc_metrics':metrics}}))
        (trial/'result.json').write_text(json.dumps({'finished_at':'now','verifier_result':{'rewards':{'reward':reward}}}))
        return job,records

    def test_summary_preserves_unknown_costs_and_relative_links_after_move(self):
        with TemporaryDirectory() as temporary:
            root=Path(temporary); self.fixture(root,unknown=True)
            before={str(p.relative_to(root)):p.read_bytes() for p in root.rglob('*') if p.is_file()}
            with patch('subprocess.run',side_effect=AssertionError('read only')),patch('urllib.request.urlopen',side_effect=AssertionError('read only')):
                first=report(root/'jobs')
            self.assertEqual(before,{str(p.relative_to(root)):p.read_bytes() for p in root.rglob('*') if p.is_file()})
            self.assertIsNone(first['summary']['total_cost_usd'])
            self.assertEqual(first['summary']['unknown_cost_attempts'],1)
            self.assertEqual(first['attempts'][0]['unknown_actor_cost_calls'],1)
            shutil.move(root/'jobs',root/'moved')
            moved=report(root/'moved');self.assertEqual(first,moved)
            for path in moved['attempts'][0]['evidence'].values():self.assertTrue((root/'moved'/path).exists())

    def test_matched_comparison_surfaces_contract_differences_and_mismatched_tasks(self):
        with TemporaryDirectory() as temporary:
            root=Path(temporary); self.fixture(root/'old',reward=0,condition='phase3_control');self.fixture(root/'new',reward=1)
            old,new=report(root/'old/jobs'),report(root/'new/jobs')
            result=compare(old,new);self.assertEqual(result['summary']['gain'],1)
            self.assertIn('consultation',result['pairs'][0]['configuration_differences'])
            new['attempts'][0]['provenance']['task_package_sha256']='changed'
            result=compare(old,new);self.assertEqual(result['summary']['eligible_scored_pairs'],0)
            self.assertIn('different task_package_sha256',result['pairs'][0]['provenance_issues'])
            old['attempts'].append(old['attempts'][0])
            with self.assertRaisesRegex(ValueError,'multiple attempts'):compare(old,new)

    def test_missing_provenance_and_corrupt_result_are_not_treated_as_comparable_success(self):
        with TemporaryDirectory() as temporary:
            root=Path(temporary); job,records=self.fixture(root,provenance=False)
            first=report(records)
            self.assertEqual(compare(first,first)['summary']['eligible_scored_pairs'],0)
            (job/'trial/result.json').write_text('{broken')
            second=report(job)
            self.assertEqual(second['attempts'][0]['status'],'incomplete_records')
            self.assertEqual(second['attempts'][0]['outcome'],'unknown')

    def test_moved_batch_uses_relative_artifact_root_and_reports_unresolved_claim(self):
        with TemporaryDirectory() as temporary:
            root=Path(temporary);job,records=self.fixture(root)
            batch=root/'jobs/_batch/cohort';batch.mkdir(parents=True)
            (batch/'metadata.json').write_text(json.dumps({'jobs_root_relative':'../..'}))
            claims=[{'task_id':job.name,'job_name':job.name,'return_code':0}, {'task_id':'unfinished','job_name':'unfinished'}]
            (batch/'claims.jsonl').write_text('\n'.join(json.dumps(c) for c in claims))
            shutil.move(root/'jobs',root/'moved')
            result=report(root/'moved/_batch/cohort')
            self.assertEqual(result['summary']['passed'],1);self.assertEqual(result['summary']['ungraded'],1)
            self.assertEqual(result['summary']['unknown_cost_attempts'],1)

    def test_cli_exports_relative_links_and_does_not_overwrite_source_files(self):
        with TemporaryDirectory() as temporary:
            root=Path(temporary);self.fixture(root)
            output=root/'reports/summary.json'
            with patch('builtins.print'):
                self.assertEqual(main(['report',str(root/'jobs'),'--format','json','--output',str(output)]),0)
                self.assertEqual(main(['report',str(root/'jobs'),'--format','json','--output',str(output)]),2)
            result=json.loads(output.read_text())
            for ref in result['attempts'][0]['evidence'].values():self.assertTrue((output.parent/ref).exists())
            self.assertIn('task_id,profile',render(result,'csv'))

    def test_missing_call_statistics_never_fabricate_zero_spend(self):
        with TemporaryDirectory() as temporary:
            root=Path(temporary);job=root/'job';(job/'agent').mkdir(parents=True)
            (job/'agent/mini-swe-agent.trajectory.json').write_text('{"info":{},"messages":[]}')
            result=report(job)
            self.assertIsNone(result['summary']['total_cost_usd'])
            self.assertIsNone(result['summary']['summed_task_duration_seconds'])


if __name__ == '__main__':unittest.main()
