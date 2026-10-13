"""Verify wheel resources and the first-use workflow outside the checkout."""
from importlib.resources import files
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def verify():
    import autoresearch_budget
    checkout=Path(__file__).resolve().parents[1]
    module=Path(autoresearch_budget.__file__).resolve()
    if module.is_relative_to(checkout):
        raise AssertionError('Verify a non-editable installed wheel, not the source checkout')
    templates=files('autoresearch_budget').joinpath('resources')
    assert json.loads(templates.joinpath('sample-24-v2.json').read_text())['tasks']
    environment=dict(os.environ)
    environment.pop('PYTHONPATH',None)
    environment.pop('OPENROUTER_KEY',None);environment.pop('OPENROUTER_API_KEY',None)
    environment.update(MSWEA_SILENT_STARTUP='1',LITELLM_LOCAL_MODEL_COST_MAP='True')
    with tempfile.TemporaryDirectory(prefix='autoresearch wheel workspace ') as temporary:
        root=Path(temporary).resolve()
        def command(*args):
            completed=subprocess.run([sys.executable,'-m','autoresearch_budget.cli',*map(str,args)],
                                     cwd=root,env=environment,capture_output=True,text=True,check=False)
            if completed.returncode:
                raise AssertionError(completed.stdout+completed.stderr)
            return completed.stdout
        command('--help');command('mock-run')
        result=json.loads(command('init','--workspace',root/'experiment','--preset','hybrid'))
        config=Path(result['config'])
        assert json.loads(command('doctor','--config',config,'--offline'))['ok']
        plan=json.loads(command('trial','plan','--config',config,'--profile','small','--task-id','HKUDS__nanobot-4048','--json'))
        assert not plan['budget_enforced']
        argv=plan['trials'][0]['command']
        assert 'autoresearch_budget.docker_runtime' in argv
        assert str(root/'experiment') in argv[argv.index('--path')+1]
        demo=root/'experiment/runs/demo'
        output=json.loads(command('demo','--output',demo));assert output['advice_status']=='validated'
        report=root/'reports/demo.json'
        command('report',demo,'--format','json','--output',report)
        result=json.loads(report.read_text())
        assert result['summary']['total_cost_usd']=='0'
        assert result['attempts'][0]['grading_status']=='unavailable'
        for reference in result['attempts'][0]['evidence'].values():
            assert (report.parent/reference).exists(), reference
    print('Installed-wheel workflow verified outside the checkout.')


if __name__=='__main__':verify()
