import os
from pathlib import Path
import signal
import subprocess
import sys
from tempfile import TemporaryDirectory
import time
import unittest

from autoresearch_budget.processes import run_command


@unittest.skipUnless(os.name=='posix','POSIX process ownership')
class ProcessTest(unittest.TestCase):
    def test_timeout_kills_host_descendants(self):
        with TemporaryDirectory() as temporary:
            root=Path(temporary);marker=root/'orphan.txt'
            program="import subprocess,time,sys; subprocess.Popen([sys.executable,'-c',sys.argv[1]]); time.sleep(20)"
            child=f"import time; from pathlib import Path; time.sleep(0.7); Path({str(marker)!r}).write_text('orphan')"
            with self.assertRaises(subprocess.TimeoutExpired):
                run_command([sys.executable,'-c',program,child],timeout=0.2)
            time.sleep(0.8)
            self.assertFalse(marker.exists())

    def test_cli_termination_stops_launched_group_and_retains_control(self):
        with TemporaryDirectory() as temporary:
            root=Path(temporary);marker=root/'orphan.txt';ready=root/'ready'
            program="""import sys,time
from pathlib import Path
from autoresearch_budget.processes import cancellation_scope,run_command
with cancellation_scope():
    Path(sys.argv[1]).write_text('ready')
    run_command([sys.executable,'-c',sys.argv[2]])
"""
            child=f"import time; from pathlib import Path; time.sleep(1); Path({str(marker)!r}).write_text('orphan'); time.sleep(20)"
            environment=dict(os.environ)
            import autoresearch_budget
            environment['PYTHONPATH']=str(Path(autoresearch_budget.__file__).resolve().parent.parent)
            process=subprocess.Popen([sys.executable,'-c',program,str(ready),child],env=environment,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            try:
                deadline=time.monotonic()+5
                while not ready.exists() and time.monotonic()<deadline:time.sleep(0.01)
                self.assertTrue(ready.exists());time.sleep(0.1)
                process.send_signal(signal.SIGTERM);process.wait(timeout=8)
                time.sleep(1.1);self.assertFalse(marker.exists())
            finally:
                if process.poll() is None:process.kill();process.wait()


if __name__=='__main__':unittest.main()
