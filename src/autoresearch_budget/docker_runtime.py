"""Run pinned Harbor with an isolated anonymous Docker client configuration."""
from __future__ import annotations

import json
import os
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import subprocess
import sys
import tempfile

HARBOR_VERSION = "0.23.0"
MINI_VERSION = "2.4.6"


def check_runtime() -> None:
    for package, expected in (("harbor", HARBOR_VERSION), ("mini-swe-agent", MINI_VERSION)):
        try:
            actual = version(package)
        except PackageNotFoundError as error:
            raise ValueError(f"Install autoresearch-budget[runtime]; missing {package}") from error
        if actual != expected:
            raise ValueError(f"{package} {actual} is unsupported; install {expected}")


def main(argv: list[str] | None = None) -> int:
    command = list(sys.argv[1:] if argv is None else argv)
    if not command:
        print("error: expected a Harbor command", file=sys.stderr)
        return 2
    try:
        check_runtime()
        # An explicitly selected binary may belong to a different environment.
        if os.environ.get("HARBOR_BIN"):
            output = subprocess.run([command[0], "--version"], check=True,
                                    capture_output=True, text=True, timeout=20)
            if HARBOR_VERSION not in output.stdout.split():
                raise ValueError(f"HARBOR_BIN must report version {HARBOR_VERSION}")
        environment = dict(os.environ)
        original = Path(environment.get("DOCKER_CONFIG", str(Path.home() / ".docker")))
        # Respect a caller's explicit daemon selection. Otherwise preserve the
        # selected context before replacing Docker's client configuration.
        host = environment.get("DOCKER_HOST")
        if not host:
            context = subprocess.run(["docker", "context", "show"], check=True,
                                     capture_output=True, text=True, timeout=20).stdout.strip()
            host = subprocess.run(["docker", "context", "inspect", context, "--format",
                                   "{{.Endpoints.docker.Host}}"], check=True,
                                  capture_output=True, text=True, timeout=20).stdout.strip()
        with tempfile.TemporaryDirectory(prefix="autoresearch-docker-") as directory:
            plugins = original / "cli-plugins"
            config = {"cliPluginsExtraDirs": [str(plugins)]} if plugins.is_dir() else {}
            Path(directory, "config.json").write_text(json.dumps(config) + "\n")
            environment.update(DOCKER_CONFIG=directory, DOCKER_HOST=host)
            environment.pop("DOCKER_CONTEXT", None)
            return subprocess.run(command, env=environment, check=False).returncode
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
