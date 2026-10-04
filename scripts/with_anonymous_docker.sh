#!/bin/sh
set -eu

if [ "$#" -eq 0 ]; then
    echo "usage: $0 COMMAND [ARG ...]" >&2
    exit 2
fi

# Docker Desktop's credential helper can block anonymous pulls. Use a temporary
# client config with no credential store while retaining the existing Compose
# plugin and daemon socket. This does not alter ~/.docker/config.json.
original_config=${DOCKER_CONFIG:-"$HOME/.docker"}
context=$(docker context show)
docker_host=$(docker context inspect "$context" --format '{{.Endpoints.docker.Host}}')
temporary_config=$(mktemp -d "${TMPDIR:-/tmp}/autoresearch-docker.XXXXXX")
trap 'rm -rf "$temporary_config"' EXIT HUP INT TERM

python3 - "$temporary_config/config.json" "$original_config/cli-plugins" <<'PY'
import json
import sys
from pathlib import Path

output = Path(sys.argv[1])
plugin_dir = Path(sys.argv[2])
config = {}
if plugin_dir.is_dir():
    config["cliPluginsExtraDirs"] = [str(plugin_dir)]
output.write_text(json.dumps(config) + "\n")
PY

DOCKER_CONFIG=$temporary_config DOCKER_HOST=$docker_host "$@"
