import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from autoresearch_budget.budget import BudgetLedger, inspect_ledger
from autoresearch_budget.cli import main
from autoresearch_budget.onboarding import initialize
from autoresearch_budget.runner import ExperimentConfig, HarborTrialRunner, TrialSpec


class PublicControlsTest(unittest.TestCase):
    def test_invalid_config_never_enables_paid_work_or_creates_outputs(self):
        with TemporaryDirectory() as temporary:
            root=Path(temporary); initialize(root,'standalone-small')
            config=root/'experiment.local.json'; baseline=json.loads(config.read_text())
            cases=[('paid_inference_enabled','false'),('budget.cap_usd','NaN'),('budget.enforce','false'),
                   ('agent.max_steps',True),('agent.command_timeout_sec',0),('profiles.small.run_max_cost_usd',-1)]
            for name,value in cases:
                raw=json.loads(json.dumps(baseline)); target=raw; parts=name.split('.')
                for key in parts[:-1]:target=target[key]
                target[parts[-1]]=value; config.write_text(json.dumps(raw))
                with patch('subprocess.run',side_effect=AssertionError('no dispatch')),patch('builtins.print'):
                    self.assertEqual(main(['trial','run','--config',str(config),'--profile','small','--task-id','test']),2)
                self.assertFalse((root/'runs').exists())

    def test_single_standalone_attempt_cannot_overwrite_existing_job(self):
        with TemporaryDirectory() as temporary:
            root=Path(temporary); initialize(root,'standalone-small'); path=root/'experiment.local.json'
            raw=json.loads(path.read_text());raw['paid_inference_enabled']=True;path.write_text(json.dumps(raw))
            config=ExperimentConfig.load(path); runner=HarborTrialRunner(config);spec=TrialSpec('example','small')
            (config.jobs_root/runner._job_name(spec)).mkdir(parents=True)
            with patch('subprocess.run',side_effect=AssertionError('no dispatch')):
                with self.assertRaisesRegex(ValueError,'already exists'):runner.run(spec)

    def test_ledger_inspection_is_read_only_and_reconciliation_retains_evidence(self):
        with TemporaryDirectory() as temporary:
            root=Path(temporary);db=root/'budget.sqlite3'
            with patch('builtins.print'):
                self.assertEqual(main(['budget','show','--db',str(db)]),2)
            self.assertFalse(db.exists())
            ledger=BudgetLedger(db,'2'); ledger.reserve('r1',run_id='run',model='mock',max_cost_usd='1')
            ledger.mark_uncertain('r1','verify provider')
            before=db.read_bytes();state=inspect_ledger(db);self.assertEqual(before,db.read_bytes())
            self.assertEqual(state['snapshot']['held_usd'],'1.000000')
            evidence=root/'billing.json';evidence.write_text('{"cost":0.25}')
            with patch('builtins.print'):
                self.assertEqual(main(['budget','reconcile','--db',str(db),'--reservation-id','r1','--actual-cost-usd','0.25',
                    '--evidence',str(evidence),'--note','Verified provider billing']),0)
            state=inspect_ledger(db)
            self.assertEqual(state['snapshot']['held_usd'],'0')
            self.assertIn('sha256=',state['reservations'][0]['note'])

    def test_drain_has_explicit_clear_without_dispatch(self):
        with TemporaryDirectory() as temporary:
            root=Path(temporary);(root/'metadata.json').write_text('{}')
            with patch('builtins.print'),patch('subprocess.run',side_effect=AssertionError('no dispatch')):
                self.assertEqual(main(['batch','drain','--batch-dir',str(root)]),0)
                self.assertTrue((root/'drain.request').exists())
                self.assertEqual(main(['batch','drain','--batch-dir',str(root),'--clear']),0)
                self.assertFalse((root/'drain.request').exists())

    def test_interrupted_dispatch_preserves_records_and_uncertain_reservation(self):
        with TemporaryDirectory() as temporary:
            root=Path(temporary); initialize(root,'standalone-small'); path=root/'experiment.local.json'
            raw=json.loads(path.read_text()); raw['paid_inference_enabled']=True; path.write_text(json.dumps(raw))
            config=ExperimentConfig.load(path);task=config.task_root/'example';task.mkdir(parents=True)
            (task/'task.toml').write_text('[agent]\n[environment]\n'); (task/'instruction.md').write_text('Fix public issue')
            (root/'.env').write_text('OPENROUTER_KEY=offline-placeholder')
            runner=HarborTrialRunner(config);spec=TrialSpec('example','small')
            with patch('autoresearch_budget.runner.run_command',side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):runner.run(spec)
            records=config.jobs_root/'_run_records'/runner._job_name(spec)
            summary=json.loads((records/'summary.json').read_text())
            self.assertEqual(summary['status'],'interrupted');self.assertIsNone(summary['total_cost_usd'])
            self.assertEqual(inspect_ledger(config.budget_db)['reservations'][0]['status'],'uncertain')
            self.assertTrue(json.loads((records/'metadata.json').read_text())['provenance']['task_package_sha256'])


if __name__ == '__main__':unittest.main()
