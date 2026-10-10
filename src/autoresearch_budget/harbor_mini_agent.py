"""Harbor mini-SWE-agent adapter with a portable uv tool path.

Harbor 0.23.0 installs the tool into ``uv tool dir --bin`` but assumes that
directory is always ``$HOME/.local/bin``. Some SWE-rebench images set XDG data
paths independently of HOME, so the installed executable is otherwise missed.
"""

from __future__ import annotations

import shlex
import hashlib
import json
from pathlib import Path

from harbor.agents.installed.mini_swe_agent import MiniSweAgent
from harbor.environments.base import BaseEnvironment


class BudgetMiniSweAgent(MiniSweAgent):
    """Use Harbor's mini-SWE implementation with a stable executable shim."""

    async def install(self, environment: BaseEnvironment) -> None:
        await self.ensure_system_dependencies(
            environment, ("curl", "bash", "build_tools", "git")
        )

        version_spec = f"=={self._version}" if self._version else ""
        python_version = shlex.quote("3.12")
        await self.exec_as_agent(
            environment,
            command=(
                "set -euo pipefail; "
                "if ! command -v uv >/dev/null 2>&1; then"
                "  curl -LsSf https://astral.sh/uv/install.sh | sh;"
                " fi && "
                'if [ -f "$HOME/.local/bin/env" ]; then '
                '  source "$HOME/.local/bin/env";'
                " fi && "
                f"uv python install {python_version} && "
                f"uv tool install --python {python_version} "
                f"mini-swe-agent{version_spec} "
                "--with litellm --with orjson --with fastapi && "
                'tool_bin="$(uv tool dir --bin)" && '
                'mkdir -p "$HOME/.local/bin" && '
                'ln -sf "$tool_bin/mini-swe-agent" '
                '"$HOME/.local/bin/mini-swe-agent" && '
                'export PATH="$HOME/.local/bin:$tool_bin:$PATH" && '
                "mini-swe-agent --help"
            ),
        )


class AdvisoryMiniSweAgent(BudgetMiniSweAgent):
    """Install the small project extension alongside pinned mini-SWE."""

    SOURCE_DIR = "/opt/autoresearch-poc"
    SOURCE_FILES = ("__init__.py", "models.py", "store.py", "contracts.py", "advisor.py", "actor_model.py", "poc_environment.py", "advisory_agent.py")

    async def install(self, environment: BaseEnvironment) -> None:
        await super().install(environment)
        target = f"{self.SOURCE_DIR}/autoresearch_budget"
        await self.exec_as_root(environment, command=f"mkdir -p {shlex.quote(target)}")
        fingerprints = {}
        for name in self.SOURCE_FILES:
            source = Path(__file__).parent / name
            fingerprints[name] = hashlib.sha256(source.read_bytes()).hexdigest()
            await environment.upload_file(source, f"{target}/{name}")
        await self.exec_as_root(environment, command=f"chmod -R a+rX {shlex.quote(self.SOURCE_DIR)}")
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        (self.logs_dir / "poc-source.json").write_text(json.dumps(fingerprints, indent=2) + "\n")

    async def exec_as_agent(self, environment, command, env=None, cwd=None, timeout_sec=None):
        # The parent launcher supplies model credentials. Its child-shell config
        # still masks those credentials, just as for the standalone baselines.
        overlay = dict(env or {})
        suffix = overlay.get("PYTHONPATH", "")
        overlay["MSWEA_GLOBAL_COST_LIMIT"] = "0"
        overlay["MSWEA_GLOBAL_CALL_LIMIT"] = "0"
        overlay["PYTHONPATH"] = self.SOURCE_DIR + (":" + suffix if suffix else "")
        return await super().exec_as_agent(environment, command, overlay, cwd, timeout_sec)

    def populate_context_post_run(self, context) -> None:
        super().populate_context_post_run(context)
        summary_path = self.logs_dir / "advice" / "summary.json"
        if summary_path.is_file():
            metrics = json.loads(summary_path.read_text())
            total = metrics.get("total_cost_usd")
            context.cost_usd = float(total) if total is not None else None
            advisor = metrics.get("advisor", {})
            context.n_input_tokens = (context.n_input_tokens or 0) + advisor.get("prompt_tokens", 0)
            context.n_output_tokens = (context.n_output_tokens or 0) + advisor.get("completion_tokens", 0)
