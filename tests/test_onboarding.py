import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import importlib.util

from autoresearch_budget.onboarding import initialize, PRESETS, doctor
from autoresearch_budget.runner import ExperimentConfig, HarborTrialRunner, TrialSpec
from autoresearch_budget.dataset import inspect_tasks
from autoresearch_budget.demo import run_demo


class OnboardingTest(unittest.TestCase):
    def test_every_preset_plans_and_paid_runs_remain_disabled(self):
        with TemporaryDirectory(prefix='onboarding space ') as temporary:
            for preset in PRESETS:
                root = Path(temporary) / preset
                result = initialize(root, preset)
                config = ExperimentConfig.load(result['config'])
                self.assertFalse(config.paid_inference_enabled)
                task = json.loads((root/'tasks.json').read_text())['tasks'][0]['task_id']
                self.assertIn('--path', HarborTrialRunner(config).command(TrialSpec(task, result['profile'])))

    def test_initialization_preserves_existing_files_and_env(self):
        with TemporaryDirectory() as temporary:
            root=Path(temporary)
            (root/'.env').write_text('PRIVATE=preserve')
            initialize(root, 'hybrid')
            before={p.name:p.read_bytes() for p in root.iterdir() if p.is_file()}
            with self.assertRaises(ValueError):initialize(root, 'scheduled')
            self.assertEqual(before,{p.name:p.read_bytes() for p in root.iterdir() if p.is_file()})

    def test_inspection_has_stable_hashes_but_no_verifier_contents(self):
        with TemporaryDirectory() as temporary:
            root=Path(temporary); task=root/'tasks/owner__repo-1'; task.mkdir(parents=True)
            (task/'task.toml').write_text('[agent]\n[environment]\n')
            (task/'instruction.md').write_text('Public issue')
            (task/'environment').mkdir(); (task/'tests').mkdir()
            (task/'tests/secret.txt').write_text('hidden solution content')
            manifest=root/'manifest.json'; manifest.write_text(json.dumps({'tasks':[{'task_id':task.name}]}))
            first=inspect_tasks(root/'tasks',manifest)
            self.assertNotIn('hidden solution content',json.dumps(first))
            self.assertEqual(first,inspect_tasks(root/'tasks',manifest))
            (task/'tests/secret.txt').write_text('changed package')
            self.assertNotEqual(first['tasks'][0]['package_sha256'],inspect_tasks(root/'tasks',manifest)['tasks'][0]['package_sha256'])
            manifest.write_text(json.dumps({'tasks':[{'task_id':'../outside'}]}))
            with self.assertRaises(ValueError):inspect_tasks(root/'tasks',manifest)

    def test_offline_doctor_does_not_launch_docker_or_network(self):
        with patch('autoresearch_budget.onboarding.check_runtime'), patch('subprocess.run',side_effect=AssertionError('no process')):
            self.assertTrue(doctor(None,offline=True)['ok'])

    @unittest.skipUnless(importlib.util.find_spec('minisweagent'), 'requires runtime extra')
    def test_demo_completes_advice_with_real_evidence_without_network(self):
        with TemporaryDirectory() as temporary, patch('urllib.request.urlopen',side_effect=AssertionError('no network')):
            output=Path(temporary)/'demo'
            result=run_demo(output)
            self.assertEqual(result['advice_status'],'validated')
            self.assertIn('+    return a + b',(output/'patch.diff').read_text())
            events=(output/'agent/advice/events.jsonl').read_text()
            self.assertIn('Regression passed',events)
            self.assertEqual(json.loads((output/'summary.json').read_text())['total_cost_usd'],'0')
            with self.assertRaises(ValueError):run_demo(output)


if __name__ == '__main__':unittest.main()
