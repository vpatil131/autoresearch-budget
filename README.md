# Autoresearch on a budget

A research harness for studying how a small coding agent uses bounded advice from a frontier model. It runs standalone and advisory agents on Harbor software-engineering tasks, records their actions and evidence, and collects independent verifier results.

The repository contains source code, tests, configuration examples, and a task-identity manifest. Experiment results, downloaded task packages, trajectories, budget ledgers, and presentation material are stored separately.

## Architecture

Harbor provides the task environment and post-submission verifier. Standalone runs use mini-SWE-agent. Advisory runs add structured actor actions, consultation policies, evidence-linked advice progress, and recovery checkpoints. OpenRouter supplies inference for the configured actor and advisor profiles.

The actor owns commands, edits, and submission. The advisor receives public evidence and returns guidance; it cannot execute commands or inspect hidden verifier results. See [architecture and runtime contracts](docs/architecture.md).

## Setup

Use Python 3.12 or newer for Harbor execution and the full runtime tests; the lightweight package supports Python 3.11. The commands below assume `python3` selects the intended version. Run them from the repository root: the current launcher resolves configuration and helper-script paths relative to that working directory. Execution uses POSIX facilities; use macOS or Linux.

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e .
```

The mock run and lightweight tests require no API key or Docker:

```sh
autoresearch-budget mock-run
python -m unittest discover -s tests -v
```

The mock run exercises the scripted shell scaffold. Runtime tests are skipped when their optional dependencies are absent. To run all tests and use the Harbor adapters, install the pinned dependencies:

```sh
python -m pip install 'harbor==0.23.0' 'mini-swe-agent==2.4.6'
MSWEA_SILENT_STARTUP=1 LITELLM_LOCAL_MODEL_COST_MAP=True \
  python -m unittest discover -s tests -v
```

Real task execution also requires Docker, its Compose plugin, and downloaded Harbor task packages. The default task location is `data/harbor/swe-rebench-07-2026`. Each selected task must have its own directory containing `task.toml`, `instruction.md`, and the package's environment/verifier files. Download task packages separately; this repository does not contain them. Set `locations.task_root` in your local configuration to use another location.

## Configuration examples

| File | Scaffold and policy |
| --- | --- |
| `config/experiment.example.json` | Standalone mini-SWE-agent with budget reservations |
| `config/experiment.phase1.v2.json` | Advisory agent with fixed-interval guidance |
| `config/experiment.phase2.json` | Actor-requested guidance |
| `config/experiment.phase3.json` | Planning and recovery guidance with the hybrid policy |
| `config/experiment.phase1.json` | Earlier fixed-interval settings, retained for compatibility |

The phase filenames are retained for compatibility. Every checked-in example disables paid inference. Model/provider settings are configurable examples. `config/sample-24-v2.json` contains frozen task identities and dataset provenance, without observed scores, costs, or run IDs. It is a selected development cohort, not a representative benchmark sample.

Standalone runs reserve their per-run allowance in a shared SQLite ledger and reconcile known final costs. Uncertain costs keep their reservation held. Advisory runs currently require `budget.enforce=false`: they record costs and enforce actor/consultation call limits, but do not enforce dollar budgets. Legacy dollar amounts in advisory configurations do not cap their spending.

## Preview a task

Planning does not launch containers or inference. Single-task plans can be inspected before downloading task packages:

```sh
autoresearch-budget trial plan \
  --config config/experiment.example.json \
  --profile small --task-id HKUDS__nanobot-4048
```

Preview advisory execution using the hybrid example:

```sh
autoresearch-budget trial plan \
  --config config/experiment.phase3.json \
  --profile small --task-id HKUDS__nanobot-4048 \
  --advice-policy plan_review --max-advice-calls 7
```

Available advice policies are `actor_only`, `fixed_interval`, `advice_requested`, `plan_review`, `recovery_review`, `hybrid`, and `phase3_control`. Zero consultations removes advice requests and automatic reviews while retaining the selected policy's local planning/checkpoint behavior. `--advice-every-steps` applies only to `fixed_interval`.

## Execute a task or cohort

Create an ignored local configuration from the desired example:

```sh
cp config/experiment.example.json config/experiment.local.json
cp -n .env.example .env
```

Set `OPENROUTER_KEY` or `OPENROUTER_API_KEY` in `.env` or the host environment. Set the local configuration's task/output locations and allowances, then explicitly set `paid_inference_enabled` to `true` to enable paid execution. Keep credentials in the local environment file.

```sh
autoresearch-budget trial run \
  --config config/experiment.local.json \
  --profile small --task-id HKUDS__nanobot-4048
```

Repeat `--task-id` for selected tasks, or pass a manifest and worker count:

```sh
autoresearch-budget trial run \
  --config config/experiment.local.json --profile small \
  --task-manifest config/sample-24-v2.json --workers 4
```

`trial` supports one to eight workers. Use a new attempt number (`--attempt`) or jobs directory to preserve prior artifacts. Multi-task execution validates task packages before dispatch and stops starting new work after a runner error; active trials finish.

For resumable advisory cohorts, create your local configuration from an advisory example and use `batch`. Its concurrency range is one to six, defaulting to four. Batch planning requires the selected task packages to be present.

```sh
autoresearch-budget batch plan \
  --config config/experiment.local.json \
  --batch-dir runs/advisory/_batch/example --concurrency 4
```

Replace `plan` with `run` to execute. Resume with the same configuration and batch directory. Changed settings require a new batch directory; unresolved attempts require reconciliation before further dispatch.

```sh
autoresearch-budget batch status \
  --batch-dir runs/advisory/_batch/example --live
```

Omit `--live` to read only host records. Creating `drain.request` inside the batch directory stops new dispatch while active tasks finish; remove it before resuming.

## Outputs and repository layout

Harbor job directories retain trajectories, patches, console logs, and grading outputs. Advisory jobs additionally retain evidence events and advice artifacts under the trial's `agent/advice/` directory. Host records live under `<jobs_root>/_poc_records/`; batch journals and source/configuration snapshots live in the selected batch directory.

- `src/autoresearch_budget/`: agents, model gateway, consultation contracts, runner, batch coordinator, ledger, and artifact store.
- `tests/`: offline unit and runtime integration tests.
- `config/`: disabled execution examples and clean task identities.
- `scripts/`: Docker helpers, smoke tests, and dataset-metadata extraction.
- `docs/`: current architecture and configuration contracts.

Generated files belong in ignored output locations. `scripts/build_candidate_manifest.py` extracts metadata from a downloaded task package and defaults to `data/manifests/july_2026_manifest.json`; it excludes oracle and test-patch bodies. Experiment archives and presentation builds are kept outside the code repository.
