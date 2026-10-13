# Execution, reporting, and recovery

## Attempts and records

Each trial uses a task/profile/attempt identity. An exclusive host claim prevents simultaneous launches from overwriting an attempt. Use a new attempt number or jobs directory for new work; never delete uncertain records merely to make the runner retry.

Harbor job directories contain trajectories, patches and verifier results. Advisory runtime evidence lives under `<job>/<trial>/agent/advice/`. Host records are under `<jobs_root>/_poc_records/<job>` for advisory attempts and `<jobs_root>/_run_records/<job>` for standalone attempts. Host metadata retains resolved settings, task-package hashes, runtime dependency versions (including their required dependency closure), source hashes, and contract identity. Grading records are collected after solving; they do not enter advisor packets.

Batch directories retain configuration, manifest, metadata, source snapshots, claims, attempts, events and status. The coordinator persists claims before dispatch. Configuration, manifest, task content and runtime identity enter the batch fingerprint. Resume with the same config, task contents, runtime and batch directory; valid attempts are skipped, terminal claims are recovered, and ambiguous claims or integration failures stop scheduling. Inspect `batch plan` and `batch status` before deciding how to reconcile an interrupted cohort.

Host launcher cancellation stops owned descendant process groups and retains partial records. Docker resources are managed separately by Harbor; inspect container state after interruption before resuming.

`batch drain` requests a graceful stop of new scheduling; active work finishes. Clear the request with `batch drain --clear` before resuming. `batch status --live` additionally reads running Docker containers; omit it for host-only inspection.

## Standalone reservations

```sh
autoresearch-budget budget show --db runs/budget.sqlite3
autoresearch-budget budget reservations --db runs/budget.sqlite3
```

Both commands read existing ledgers without creating them. Active and uncertain reservations consume capacity. A timeout or missing final cost is not proof of zero billing. Verify provider billing or authoritative completed-cost evidence before explicitly settling a hold:

```sh
autoresearch-budget budget reconcile --db runs/budget.sqlite3 \
  --reservation-id 'harbor-mini:small-owner__repo-123-a1' \
  --actual-cost-usd 0.123 --evidence /path/to/billing-evidence.json \
  --note 'Verified final provider billing for this reservation'
```

The command records the evidence path and SHA-256 with the note. Keep the evidence file with the private experiment bundle; the CLI does not independently verify a provider statement. `budget correct` uses the same arguments to correct an already settled amount. If actual spend exceeds the reserved amount, the ledger records it and reports the discrepancy. There is no automatic cancellation of uncertain holds.

Advisory execution has observational accounting and call limits, with no combined dollar enforcement. It does not create standalone reservations. Legacy dollar fields in older advisory configs remain inert.

## Reports and comparisons

`report` accepts a jobs root, individual Harbor job, host attempt record, batch directory, or demo output. It reads local files without inference, grading, containers or modification of original artifacts. Save text, JSON or CSV to a new output file; existing output files are preserved. `--budget-db` attaches the current standalone ledger state to JSON reporting.

The version-1 JSON schema includes execution/grading status, reward, stop reason, actor usage, logical/physical advisor usage, unknown-cost counts, timing, configuration, provenance and evidence links. A zero reward remains a benchmark failure, independently of infrastructure classification. Missing scores remain ungraded; unknown costs remain null. The reported known-cost subtotal covers only attempts with complete final costs, not all spend for incomplete attempts.

Timing distinguishes host duration, Harbor agent execution, environment setup, grading, per-task end-to-end duration, summed task duration and batch wall time. A sum of task durations is not concurrent wall time. Missing timing stays unavailable. Batch wall time spans its recorded start to terminal status, so it includes gaps between resumptions.

Move the entire jobs root to retain sibling records, batches and evidence. Batch metadata stores the relative jobs root; report exports resolve evidence relative to the output file's parent. Preserve that structure when copying bundles. Regenerate reports after moving a bundle. Historical custom batches without a relative root may require their original layout; new records do not rely on stored absolute claim paths for reporting.

`compare LEFT RIGHT` requires one attempt per task per condition. It lists unmatched tasks and excludes ungraded/incomplete attempts, missing identities, or differing task-package/runtime identities from scored aggregates. It shows configuration differences and gains, regressions, shared passes and shared failures. Comparisons are descriptive: matched IDs alone do not establish causality or identical experimental conditions.

## Troubleshooting

| Symptom | Action |
| --- | --- |
| Missing or incompatible runtime | Install `.[runtime]` or the locked contributor environment; run `doctor --offline`. |
| Docker/Compose unavailable | Start Docker and verify `docker info` and `docker compose version`. |
| Task package incomplete | Point `task_root` at the parent of task directories; run `dataset inspect` on the selected manifest. |
| Attempt already exists | Inspect its records; select a new attempt/output root for genuinely new work. |
| Batch settings changed | Use a new batch directory, or resume with the original frozen settings. |
| Unresolved batch claim | Inspect host summary, Harbor process/container state, verifier records and billing before reconciliation; the runner will not silently retry it. |
| Credential missing | Set one supported key in the selected workspace environment file or host environment; doctor checks only its presence. |
| Provider/route failure | Check model availability and route settings; paid probes require explicit enabling and allowance. |
| Unknown cost or exhausted capacity | Inspect reservations and final provider evidence; reconcile explicitly rather than deleting the ledger. |
| Image architecture/resources | Ensure the task images support the host Docker architecture and the task's configured CPU/memory/disk requirements. |

The packaged Docker wrapper uses a temporary anonymous client configuration while retaining the chosen daemon and Compose plugin directory. It does not change the user's Docker configuration; private-image authentication may require a different execution setup. Supported task workflows currently assume publicly pullable SWE-rebench images.
