# Architecture and runtime contracts

The experiment runner combines Harbor task environments with standalone or advisory coding agents. Model profiles configure OpenRouter routes; task manifests select task identities. Experiment measurements and raw run artifacts are external to the code repository.

## Components

| Component | Responsibility |
| --- | --- |
| CLI and runner | Validate selection/configuration, stage tasks, launch Harbor, and collect results |
| Standalone agent | Run pinned mini-SWE-agent against the configured model |
| Advisory actor | Diagnose, propose plans, edit, execute checks, handle advice, and submit |
| Controller | Validate actions, apply consultation policy, track evidence and advice obligations |
| Advisor | Return structured guidance from public evidence; perform no repository actions |
| Harbor environment | Execute repository commands and retain their outputs |
| Harbor verifier | Grade the submitted repository after solving finishes |
| Artifact store | Record payloads, events, patch snapshots, usage, and costs |
| Budget ledger | Reserve and reconcile standalone inference allowances |

## Consultation policies

| Policy | Automatic guidance | Actor-requested guidance |
| --- | --- | --- |
| `actor_only` | None | No |
| `fixed_interval` | After each configured interval, before the final actor call | No |
| `advice_requested` | None | Yes |
| `plan_review` | Plan and inspection reviews | Yes |
| `recovery_review` | Recovery reviews | Yes |
| `hybrid` | Plan, inspection, and recovery reviews | Yes |
| `phase3_control` | None | Yes |

`max_calls` bounds logical consultations, including dispatched failures. Physical provider requests, retries, and fallback requests are recorded separately. Setting `max_calls=0` disables all consultations and advice tools. Planning and local checkpoints follow the selected policy even when advice is disabled. `phase3_control` with zero consultations is a planning control without advice.

Consultation thresholds and context/output limits are defined by `AdvicePolicy` in `advisor.py`. Recovery-enabled policies additionally bound automatic recovery calls with `max_automatic_recovery_calls`.

## Evidence and follow-through

Actor tools carry structured state and evidence references. The controller checks observation, patch, and advice IDs against recorded evidence. Guidance is acknowledged through an explicit decision, then followed through with recorded progress and command evidence. Acknowledgment alone cannot validate advice or permit submission with unresolved obligations.

Plan changes link the previous plan and explain their evidence. Follow-up consultations may refer to acknowledged guidance still open. Recovery revisions can supersede related obligations; superseded guidance remains in the record and is not counted as validated.

Automatic recovery while advice is open requires an eligible completed behavioral check after the latest open guidance. Setup/read commands, timeouts, environment blockers, checks combined with edits, and status-masking command wrappers do not qualify. Eligibility is conservative execution evidence; it does not establish that a check covers the required behavior. Semantic labels remain actor claims.

## Isolation and grading

The runner stages an attempt-specific copy of each task. Standalone/advisory mini-SWE inference runs inside Harbor with agent-phase egress limited to `openrouter.ai`. The model process receives inference credentials; configured child-shell overlays mask the OpenRouter keys from repository commands. The historical budget-shell scaffold uses a no-network agent phase.

The advisor receives the public issue, visible observations, candidate patches, structured state, and bounded transcript context. Hidden tests and verifier results are excluded from solving evidence. The runner collects grading records after Harbor exits and stores them separately under `post_submission/`.

## Execution, accounting, and resumption

Standalone mini-SWE runs reserve the configured maximum in a shared SQLite ledger before dispatch. Completed costs reconcile reservations; interrupted or unpriced calls retain conservative holds. The ledger records known spend separately from held capacity.

Advisory execution currently requires `budget.enforce=false` and a zero mini-SWE cost limit. Actor and consultation call limits still apply, but dollar caps and ledger reservations do not. Usage summaries preserve unknown costs instead of treating them as zero known spend.

`trial` schedules independent attempts with up to eight workers. Advisory `batch` execution supports up to six workers and persists claims before dispatch, then collects terminal records. Runner/integration failures stop new scheduling while active work finishes; benchmark zero rewards remain valid outcomes.

Batch configuration, task selection, package contents and runtime identity are fingerprinted; source files are snapshotted for dispatch. Resumption skips valid completed tasks, reconciles terminal claims, and rejects changed settings or ambiguous attempts. `batch drain` writes a `drain.request` flag to stop new scheduling without terminating active attempts. Host claims protect all scaffolds from duplicate attempt dispatch, and host metadata records task/package/runtime provenance.

## Extending configuration

Model profiles specify model IDs, provider routing, reasoning, service tier, and per-run amounts. Advisory fallback routes may change provider, tier, and output-allowance parameter while preserving the model request contract.

Task manifests need a `tasks` array of distinct `task_id` values. Each ID names a downloaded Harbor task directory. The included manifest also retains repository, language, commit, image, and dataset provenance; observed outcomes and historical run identifiers are excluded.

Policy changes may require corresponding actor tools, validation, evidence collectors, and offline runtime tests. Sandbox-installed advisory modules are explicitly listed in `AdvisoryMiniSweAgent.SOURCE_FILES`; dependency coverage is checked by the sandbox-package test.

## Installed workflow and reporting

Packaged templates support `init`; `doctor` validates the execution environment without inference. `demo` injects scripted transports into the production advisory controller and runs a tiny public regression fixture. The Docker wrapper is an installed Python module, and batch snapshots copy actual runtime modules. Configurations with `workspace` anchor their relative paths to the config file.

`dataset inspect` validates external task packages and records aggregate content hashes. `report` reads host/Harbor artifacts with relative evidence links; `compare` requires unique matched task identities and surfaces configuration and provenance differences. Neither command executes tasks or makes inference requests. See [configuration](configuration.md) and [execution/reporting](usage.md).
