# Autoresearch on a budget

[![Offline verification](https://github.com/vpatil131/autoresearch-budget/actions/workflows/ci.yml/badge.svg)](https://github.com/vpatil131/autoresearch-budget/actions/workflows/ci.yml)

An experimental research harness for studying how a small coding actor uses bounded guidance from a frontier advisor. Researchers can run matched standalone, advisory, and control conditions, inspect advice follow-through, and compare independently graded outcomes.

**SWE-rebench through Harbor is the initially supported benchmark.** Other compatible Harbor software engineering task packages are experimental. Direct execution on an arbitrary local repository and a general agent SDK are later capabilities.

The supported paid workflow uses **OpenRouter** for actor and advisor inference. Model profiles select models and underlying providers through that gateway. Direct provider APIs and local inference endpoints require implementation work; offline inspection, planning, reporting, and the scripted demo need no gateway credentials.

The repository contains code, tests, documentation, disabled presets, and clean task identities. Downloaded datasets, experiment results, trajectories, ledgers, and presentations belong outside tracked source.

## Install and see advice working

Use Python 3.12 or newer on macOS or Linux. Install the versioned wheel from the [GitHub release](https://github.com/vpatil131/autoresearch-budget/releases/tag/v0.1.0):

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install 'autoresearch-budget[runtime] @ https://github.com/vpatil131/autoresearch-budget/releases/download/v0.1.0/autoresearch_budget-0.1.0-py3-none-any.whl'
```

Alternatively, clone the repository and install from source:

```sh
git clone https://github.com/vpatil131/autoresearch-budget.git
cd autoresearch-budget
python3 -m venv .venv
. .venv/bin/activate
python -m pip install '.[runtime]'
```

Then create an experiment workspace and run the offline demonstration:

```sh
autoresearch-budget init --workspace ./experiment --preset hybrid
cd experiment
autoresearch-budget doctor --offline --config experiment.local.json
autoresearch-budget demo --output runs/demo
autoresearch-budget report runs/demo
```

The demo uses the current advisory controller, scripted actor/advisor responses, and a tiny temporary Git repository. It inspects a bug, proposes a plan, receives guidance, edits the file, executes a regression check, records validated advice, and submits. It needs Git but no key, Docker, task download, or inference. Its assertion is a demonstration, so reporting marks it ungraded rather than presenting a benchmark score.

For lightweight commands and tests, install `.` without the runtime extra. The advisory demo and real execution need `harbor==0.23.0` and `mini-swe-agent==2.4.6`, included in `[runtime]`. Contributor installations use the checked-in `uv.lock`; see [CONTRIBUTING.md](CONTRIBUTING.md).

Initialization preserves existing files and leaves paid execution disabled. New configs pin their workspace relative to the config file, allowing commands from unrelated directories. [Path rules and configuration](docs/configuration.md) describe legacy compatibility.

## Select a condition

| Preset | Scaffold / consultation policy | Actor profile |
| --- | --- | --- |
| `standalone-small` | mini-SWE-agent with reserved accounting | `small` |
| `standalone-frontier` | mini-SWE-agent with reserved accounting | `frontier` |
| `scheduled` | Advice every 10 actor calls | `small` |
| `requested` | Actor-requested advice | `small` |
| `plan-review` | Plan and inspection review | `small` |
| `recovery-review` | Recovery review | `small` |
| `hybrid` | Plan, inspection, and recovery review | `small` |
| `control` | Planning scaffold with zero consultations | `small` |

Generate each condition in a separate workspace or use a distinct `locations.jobs_root`. The profiles in the templates are configurable examples; verify current model/provider availability before paid work.

**Advisory runs do not enforce dollar caps.** They require `budget.enforce=false`, record observed costs, and enforce actor/consultation call limits. Generated advisory presets omit inert dollar allowances. Unknown costs remain unknown. Standalone runs reserve allowances in a shared ledger and retain holds when final costs are uncertain.

## Run one SWE-rebench task

Obtain Harbor task packages separately, then set `locations.task_root` to their parent directory. The included development cohort identifies `ibragim-badertdinov/swe-rebench-07-2026`, version `1.0.0`, revision `1`; it is a selected 24-task cohort, not a representative benchmark sample. Preserve the dataset source/version and package checksums. Image tags and provider routes can change independently of package contents.

Each selected task needs `task.toml`, `instruction.md`, `environment/`, and `tests/`. Validate an existing external task root and save its content identities:

```sh
autoresearch-budget dataset inspect \
  --task-root /path/to/swe-rebench-tasks \
  --task-manifest tasks.json --output runs/task-identities.json
```

The command hashes package contents without printing hidden verifier or solution material. Configure the same task root in `experiment.local.json`. The harness currently supports validated external packages; it does not provide a hosted dataset downloader.

Preview a task without launching Docker, grading, or inference:

```sh
autoresearch-budget trial plan --config experiment.local.json \
  --profile small --task-id HKUDS__nanobot-4048 --json
```

For live execution, Docker with its Compose plugin must be running. Copy `.env.example` to `.env` without overwriting existing credentials. Set `OPENROUTER_API_KEY` or `OPENROUTER_KEY` there or in the host environment. Review model/provider routes and limits, then explicitly set `paid_inference_enabled` to `true` in the local config.

```sh
autoresearch-budget doctor --config experiment.local.json
autoresearch-budget trial run --config experiment.local.json \
  --profile small --task-id HKUDS__nanobot-4048
autoresearch-budget report runs/hybrid
```

`doctor` checks configuration, runtime versions, Docker/Compose, credential presence, and advisory task packages. It never makes a model request or prints key values. Standalone configurations without a consultation manifest check the task root; inspect the selected manifest separately before dispatch. Use `--env-file` if credentials live elsewhere.

## Run and compare cohorts

`trial` accepts repeated `--task-id` or `--task-manifest`, with one to eight workers. Advisory `batch` adds resumption and supports one to six concurrent tasks. Plans expose selected routes, limits, and accounting. Batch plans require task packages to be present.

```sh
autoresearch-budget batch plan --config experiment.local.json \
  --batch-dir runs/hybrid/_batch/cohort --concurrency 4
autoresearch-budget batch run --config experiment.local.json \
  --batch-dir runs/hybrid/_batch/cohort --concurrency 4
autoresearch-budget batch status --batch-dir runs/hybrid/_batch/cohort
```

Resume using the same settings and directory. Changed settings, task contents, or runtime identity need a new batch directory; unresolved attempts block dispatch. To stop new dispatch while active attempts finish:

```sh
autoresearch-budget batch drain --batch-dir runs/hybrid/_batch/cohort
# Before resuming:
autoresearch-budget batch drain --batch-dir runs/hybrid/_batch/cohort --clear
```

Use a new `--attempt` or jobs directory to preserve existing attempts. Benchmark reward zero is a completed graded outcome; runner/provider/infrastructure failures are recorded separately. See [execution and recovery](docs/usage.md).

Run the same task identities under your comparison conditions, then report or compare their output roots:

```sh
autoresearch-budget report runs/hybrid --format json --output runs/hybrid-report.json
autoresearch-budget report runs/hybrid --format csv --output runs/hybrid-report.csv
autoresearch-budget compare runs/control runs/hybrid --output runs/comparison.json
```

Reports distinguish outcome, grading status, runner status, known costs, unknown costs, calls, and timing. JSON evidence links are relative to the saved report or inspected bundle. Comparisons reject ambiguous multiple attempts, show unmatched tasks and configuration differences, and exclude ungraded or incompatible task/runtime identities from aggregates. These are descriptive comparisons; differences in prompting, limits, routing, or controller behavior do not establish an advisor's causal effect.

## Learn and extend

- [Architecture and runtime contracts](docs/architecture.md): controller, isolation, evidence, accounting, and source snapshots.
- [Configuration and extension examples](docs/configuration.md): model profiles, custom cohorts, policies, and path rules.
- [Execution, reporting, and recovery](docs/usage.md): records, holds, reconciliation, and troubleshooting.
- [Contributing and verification](CONTRIBUTING.md): locked environment, offline tests, wheel verification, and artifact handling.

Reproducing an experiment setup does not guarantee identical model outputs or scores. Historical runs require their corresponding runtime revision and archived evidence; those artifacts are distributed separately from this code.

## Citation and attribution

If you use this harness in research, cite the software and include its repository URL and the released version or exact Git commit used:

> Patil, Vashishtha (2026). *Autoresearch on a budget* [Software]. https://github.com/vpatil131/autoresearch-budget

Machine-readable software metadata is available in [CITATION.cff](CITATION.cff), also used by GitHub's “Cite this repository” feature.

The first experimental release is `0.1.0`. Record the task cohort, dataset revision, configuration, and runtime provenance alongside the software citation so others can repeat the setup.

This harness builds on the following projects:

- **[Harbor](https://github.com/harbor-framework/harbor)** provides container environments, task execution, and independent grading. The pinned runtime is `0.23.0`, licensed under [Apache-2.0](https://github.com/harbor-framework/harbor/blob/main/LICENSE). Use Harbor's [software citation](https://github.com/harbor-framework/harbor/blob/main/CITATION.cff) when reporting experiments that use it.
- **[mini-SWE-agent](https://github.com/SWE-agent/mini-swe-agent)** provides the standalone coding agent and the agent loop extended by the advisory controller. The pinned runtime is `2.4.6`, licensed under [MIT](https://github.com/SWE-agent/mini-swe-agent/blob/main/LICENSE.md). Credit the upstream project and record its version.
- **[SWE-rebench](https://swe-rebench.com/)** supplies the initially supported benchmark. Cite [SWE-rebench: An Automated Pipeline for Task Collection and Decontaminated Evaluation of Software Engineering Agents](https://arxiv.org/abs/2505.20411) (2025), and [SWE-rebench V2: Language-Agnostic SWE Task Collection at Scale](https://arxiv.org/abs/2602.23866) (2026) for V2 tasks. Credit the Harbor dataset publisher, `ibragim-badertdinov`, for the selected `swe-rebench-07-2026` package, and retain its version and revision from the task manifest.

Upstream dependencies, downloaded datasets, container images, and task repositories retain their own licenses and notices. Preserve those notices when redistributing their material; citation complements the applicable license requirements.

## License

The harness code is licensed under the [MIT License](LICENSE). The upstream materials credited above retain their own licenses.
