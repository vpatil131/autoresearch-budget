import asyncio
import json
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from autoresearch_budget.models import Completion, ScriptedModel
from autoresearch_budget.probe import run_probe
from autoresearch_budget.runner import ExperimentConfig


class ProbeTest(unittest.TestCase):
    def _config(self, root: Path) -> ExperimentConfig:
        raw = {
            "budget": {"ledger": "runs/budget.sqlite3", "cap_usd": "1.00"},
            "agent": {
                "max_steps": 2,
                "max_tokens_per_call": 100,
                "command_timeout_sec": 10,
                "max_observation_chars": 1000,
            },
            "profiles": {
                "small": {
                    "model": "mock/small",
                    "run_max_cost_usd": "0.20",
                }
            },
            "paid_inference_enabled": False,
        }
        path = root / "config.json"
        path.write_text(json.dumps(raw))
        return ExperimentConfig.load(path, project_root=root)

    def test_scripted_probe_records_valid_action_and_cost(self):
        class CostedBackend(ScriptedModel):
            async def complete(self, messages, *, request_id, max_tokens):
                base = await super().complete(
                    messages, request_id=request_id, max_tokens=max_tokens
                )
                return Completion(
                    **{
                        **base.__dict__,
                        "cost_usd": Decimal("0.001"),
                        "prompt_tokens": 10,
                        "completion_tokens": 5,
                    }
                )

        with TemporaryDirectory() as temp:
            root = Path(temp)
            result = asyncio.run(
                run_probe(
                    self._config(root),
                    profile_name="small",
                    max_cost_usd=Decimal("0.01"),
                    max_tokens=20,
                    backend=CostedBackend(
                        ['{"action":"finish","summary":"compatible"}']
                    ),
                )
            )
            self.assertTrue(result.action_format_valid)
            self.assertEqual(result.cost_usd, Decimal("0.001"))
            self.assertTrue((Path(result.run_dir) / "summary.json").exists())


if __name__ == "__main__":
    unittest.main()

