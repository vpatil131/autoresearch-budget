#!/bin/sh
set -eu

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
    echo "usage: $0 TASK_DIRECTORY [JOB_STEM]" >&2
    exit 2
fi

task_dir=$1
job_stem=${2:-"$(basename "$task_dir")-$(date -u +%Y%m%dT%H%M%SZ)"}
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
jobs_dir=${HARBOR_SMOKE_JOBS_DIR:-runs/harbor-smoke}

for agent in nop oracle; do
    "$script_dir/with_anonymous_docker.sh" \
        uvx harbor run \
        --path "$task_dir" \
        --agent "$agent" \
        --n-concurrent 1 \
        --jobs-dir "$jobs_dir" \
        --job-name "$job_stem-$agent" \
        --yes
done
