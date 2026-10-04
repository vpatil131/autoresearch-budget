"""Harbor mini-SWE-agent adapter with a portable uv tool path.

Harbor 0.23.0 installs the tool into ``uv tool dir --bin`` but assumes that
directory is always ``$HOME/.local/bin``. Some SWE-rebench images set XDG data
paths independently of HOME, so the installed executable is otherwise missed.
"""

from __future__ import annotations

import shlex

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
