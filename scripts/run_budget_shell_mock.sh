#!/bin/sh
set -eu

if [ "$#" -ne 1 ]; then
    echo "usage: $0 TASK_DIRECTORY" >&2
    exit 2
fi

task_dir=$1
project_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
job_name="budget-shell-mock-$(date -u +%Y%m%dT%H%M%SZ)"
budget_db="$project_root/runs/budget-shell-smoke.sqlite3"
responses='["{\"action\":\"shell\",\"command\":\"git status --short\"}","{\"action\":\"finish\",\"summary\":\"Zero-network Harbor adapter smoke completed.\"}"]'

export PYTHONPATH="$project_root/src${PYTHONPATH:+:$PYTHONPATH}"

run_harbor() {
    if [ -n "${HARBOR_BIN:-}" ]; then
        "$project_root/scripts/with_anonymous_docker.sh" "$HARBOR_BIN" "$@"
    elif command -v harbor >/dev/null 2>&1; then
        "$project_root/scripts/with_anonymous_docker.sh" harbor "$@"
    else
        "$project_root/scripts/with_anonymous_docker.sh" uvx harbor==0.23.0 "$@"
    fi
}

run_harbor run \
    --path "$task_dir" \
    --agent autoresearch_budget.harbor_agent:BudgetShellAgent \
    --model mock/scripted \
    --ak "budget_db=$budget_db" \
    --ak budget_cap_usd=1.00 \
    --ak run_max_cost_usd=0.10 \
    --ak max_steps=3 \
    --ak allow_paid=false \
    --ak "scripted_responses=$responses" \
    --n-concurrent 1 \
    --jobs-dir "$project_root/runs/harbor-adapter-smoke" \
    --job-name "$job_name" \
    --yes

