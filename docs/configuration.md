# Configuration and extension examples

`init` copies packaged templates into a writable experiment workspace. It never overwrites existing files, changes `.env`, or enables paid inference. The clean `tasks.json` selects the development cohort; replace it with the desired cohort before execution.

## Paths

A new generated `experiment.local.json` contains `"workspace": "."`. That value resolves relative to the configuration file. `locations.task_root`, `locations.jobs_root`, `budget.ledger`, `consultation.task_manifest`, and `--env-file` resolve relative to that workspace unless absolute.

For a config under `config/` with the workspace one directory above, use `"workspace": ".."`. Legacy configs without `workspace` retain invocation-directory path resolution. The Python `ExperimentConfig.load(..., project_root=...)` argument explicitly overrides both behaviors.

CLI paths such as `--config`, `--task-manifest` on `trial`, `--batch-dir`, `--output`, and `--db` resolve from the invocation directory. Installed templates and Docker helpers resolve from the installed package, independently of the workspace. `HARBOR_BIN` optionally selects a compatible Harbor executable; its reported version and the installed runtime versions are checked before execution. The default uses Harbor in the harness's interpreter; there is no implicit `uvx` installation fallback.

## Settings

| Section | Meaning |
| --- | --- |
| `paid_inference_enabled` | A JSON boolean; defaults to false. Paid trial, batch and probe dispatch requires true. |
| `budget.enforce` | True for standalone execution; false required for advisory execution. |
| `budget.ledger`, `cap_usd` | Standalone SQLite reservation ledger and total capacity. Advisory presets omit `cap_usd`. |
| `locations` | Task-package parent and output root. Generated outputs belong in ignored directories. |
| `agent.scaffold` | `mini-swe-agent`, `mini-swe-advisor`, or the historical `budget-shell`. |
| `agent.version` | Pinned mini-SWE-agent version `2.4.6`. |
| `agent.max_steps` | Positive actor call allowance. |
| `agent.max_tokens_per_call` | Positive output token allowance per actor request. |
| `agent.command_timeout_sec` | Positive command timeout; advisory commands also respect remaining agent time. |
| `agent.max_observation_chars` | Visible observation limit, at least 100 characters. Full command evidence is retained separately. |
| `profiles` | Named actor/advisor model routes, reasoning settings, and standalone per-run allowances. |
| `consultation` | Advisory policy, manifest, limits, advisor profile, and recovery thresholds. |

Amounts must be finite and nonnegative; call/time/token settings use JSON integers. Boolean strings such as `"false"` are rejected. Planning validates settings without inference or execution state. Real execution performs additional credential, package and attempt checks.

## Change a model route

Edit a named profile, keeping the rest of the execution condition frozen. For example:

```json
{
  "model": "your-provider/your-model",
  "provider": {
    "only": ["YourProvider"],
    "allow_fallbacks": false,
    "require_parameters": true
  },
  "run_max_cost_usd": "0.50"
}
```

The names above are placeholders; substitute an available OpenRouter model/provider. All supported paid CLI paths use OpenRouter; `provider.only` selects an upstream provider through that gateway. Direct OpenAI, Anthropic, DeepInfra, and local model endpoints require additional adapters, credential handling, network policy, and accounting integration. Changing a model string alone does not add another gateway. `run_max_cost_usd` applies to standalone reservations and the mini-SWE cost limit. Omit it for advisory presets. Optional `reasoning_effort` and `service_tier` are sent to the configured route. A paid `probe` is an explicit compatibility check, never part of `doctor`; use an enforced standalone config and an explicit `--max-cost-usd`.

Advisor fallback may only change provider, service tier, and `max_tokens_parameter`. Both routes must explicitly select providers with `allow_fallbacks=false`; the fallback cannot request Flex processing. Retry/fallback physical requests remain distinct from logical consultation limits.

## Supply a custom cohort

A minimal manifest is:

```json
{
  "source": {"harbor_dataset": "your-dataset", "version": "your-pinned-version"},
  "tasks": [{"task_id": "owner__repo-123"}, {"task_id": "owner__repo-456"}]
}
```

Task IDs must be distinct plain directory names under `locations.task_root`. Advisory membership follows `consultation.task_manifest`; it is not restricted to the bundled 24 tasks. Standalone `trial --task-manifest` selects the same IDs. Use `dataset inspect` to validate the package layout and save hashes. Additional repo, base-commit, language, image and dataset metadata are useful provenance, but measured outcomes do not belong in a task-identity manifest.

SWE-rebench is the first supported benchmark. Compatible packages from other Harbor datasets are experimental and may need image/environment adaptation. Package checksums identify file contents; pin container image digests separately when exact image reproduction is required.

## Change the consultation condition

Set `consultation.policy` to one of `actor_only`, `fixed_interval`, `advice_requested`, `plan_review`, `recovery_review`, `hybrid`, or `phase3_control`. Use `every_steps` only with `fixed_interval`. `max_calls` is the maximum logical consultations, including dispatched failures. Zero disables advisor requests while retaining the selected policy's local planning/checkpoints.

Overrides are explicit and do not edit the input config:

```sh
autoresearch-budget trial plan --config experiment.local.json --profile small \
  --task-id owner__repo-123 --advice-policy plan_review --max-advice-calls 3 --json
```

The complete effective configuration is saved with real attempts and batches. Batch resumption rejects changed condition settings. Thresholds such as `stall_steps`, `inspection_steps`, `failure_threshold`, `uncertainty_steps`, `contradiction_steps`, and `min_actor_calls_remaining` are defined and validated by `AdvicePolicy`. `max_automatic_recovery_calls` additionally bounds automatic recovery consultations; explicit follow-up requests share the total ceiling.

## Extend the engine

`models.CompletionBackend` is the internal model boundary: an async `complete(messages, *, request_id, max_tokens)` returns `Completion` with usage and raw response evidence. Tests inject scripted transports while retaining the production controller and gateway accounting. `demo.py` demonstrates this integration. These interfaces are experimental, not a stable public SDK.

A new policy needs coordinated changes to `AdvicePolicy`, actor tool schemas and prompts in `contracts.py`, dispatch/validation in `AdvisoryAgent`, and any collectors in `interventions.py` or `recovery_evidence.py`. Add offline runtime coverage for triggering, denial, zero consultations, advice acknowledgment, execution evidence, and terminal follow-through. Preserve separation between public solving evidence and hidden grading.

The Harbor advisory adapter uploads an explicit list of modules from `AdvisoryMiniSweAgent.SOURCE_FILES`. If a sandbox module gains a local dependency, update that list and keep `test_advisor_sandbox_package.py` passing. Host-only reporting, onboarding and dataset utilities do not need uploading. Batch source snapshots come from actual runtime modules rather than the experiment workspace's source tree.
