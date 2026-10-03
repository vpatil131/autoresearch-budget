# Autoresearch on a budget

Small models drive a software experimentation loop and consult a frontier model for bounded advice. The first milestone is a reproducible SWE-rebench subset of 10–20 tasks with a measured small/frontier performance gap.

## Status

Research completed on October 3, 2026. No paid model calls or benchmark executions have been made. The experiment runner and empirically validated subset are not implemented yet. The presentation's illustrative results are not experimental evidence for this project.

The local `.env` uses `OPENROUTER_KEY`; the eventual runner must accept it and supply the `OPENROUTER_API_KEY` expected by the agent library without logging either value or passing credentials to task containers.

## Findings

- OpenRouter's public catalog lists `openai/gpt-6.1-sol`. Use medium reasoning. Its catalog prices are $2/M input and $10/M output tokens before caching and other adjustments.
- The requested `qwen/qwen3.8-35b-a3b` is absent from the catalog. `qwen/qwen3.6-35b-a3b` is available and matches the talk. Clarification is pending; do not silently substitute it.
- SWE-rebench's Hugging Face leaderboard dataset has 860 unique tasks in `test`, duplicated across named monthly splits. Do not concatenate those splits with `test` as independent tasks. The latest July 2026 evaluation distribution is in Harbor instead.
- Harbor's July snapshot advertises 111 hard tasks across Go, Java, Python, Rust, and TypeScript, version 1.0.0, revision 1. It is a snapshot label, not a guarantee that tasks were created in July.
- The public leaderboard's May 15–July 1, 2026 window reports Qwen3.6-35B-A3B at 24.7% mean resolved rate and GPT-5.6 Sol medium at 62.3%. These are earlier-model aggregate results, not evidence for GPT-6.1 or individual task selection. Python alone has a 14% vs 48% gap in this window, but only 20 tasks; a Python-only pool may not yield enough stable differential tasks.
- The inspected page payload exposes task identities and aggregate statistics, not individual task verdicts. No reusable exact-pair outcome matrix has been established.
- Docker is running locally on ARM64. Published older evaluation images are x86_64. Prefer an x86 Linux worker for predictable execution time; test local emulation only with a small smoke run.
- Official GPT-6.1 guidance requires Responses for native tool calling. mini-SWE-agent v2 supports `openrouter_response` and also text actions, so endpoint compatibility must be tested before choosing the final adapter.

## Proposed implementation

Use mini-SWE-agent as the common actor scaffold, with a small project wrapper for dataset selection, budget accounting, advice checkpoints, and result collection. For the July tasks, use Harbor's task environments and verifiers. For older Hugging Face tasks, use the SWE-rebench SWE-bench fork. Keep those evaluation paths explicit rather than treating their task formats as interchangeable.

Pin the agent version, task distribution revision, repository commit, prompts, reasoning settings, provider routing, and image digest. Record patches, trajectories, token usage including reasoning and cached tokens, actual billed cost, test outcomes, elapsed time, infrastructure errors, and advisor calls. Use resumable task/run IDs and reserve worst-case request cost before issuing a call so concurrent runs cannot overspend.

Expose a common shell-action contract to both models. Use Responses when native Sol tool calls require it, or the same text-action interface for both after testing compatibility. The advisor gets a compact evidence packet and returns diagnosis or a suggested experiment; the small actor owns edits and execution.

Agent containers receive the issue, base repository, and public tests. Oracle patches, benchmark test patches, verifier scripts, solution-bearing repository history, and benchmark verdicts stay outside the actor/advisor context. Audit prebuilt images for these artifacts. Use final independent grading for selection and reporting, while controller triggers use only actor-visible evidence. Verify both that the unmodified repository fails required issue tests and that the gold solution passes with regression tests preserved.

## Proposed discovery sequence

1. Resolve the small-model identity and an overall spending cap. Test one short request per model, then 2–3 complete task runs. Validate endpoint behavior, cost logging, patch extraction, and grading before scaling.
2. Inspect the July task manifest, filter for reproducible CPU environments and clear instructions, and measure evaluator runtime with gold checks. Favor cheap tests and diverse repositories. Avoid selecting only tiny gold patches as a proxy for difficulty. Keep a manifest of all exclusions and deduplicate by task ID/base commit.
3. Screen 20 candidates first. Run the small model once. Run Sol on valid small-model failures first, then expand in batches of 10–20 as the budget permits. Infrastructure failures do not count as model failures. This sequence reduces frontier spending during discovery.
4. Repeat both standalone baselines twice more on promising differential tasks, from clean workspaces and under the same prompts, tools, and step/time limits. A provisional eligibility rule is small model 0/3 solved and Sol at least 2/3 solved. Three runs provide a screening estimate, not a precise probability. Retain failure categories and cost distributions.
5. Freeze 10–20 eligible tasks, ideally 15–20, spanning several repositories and failure types. If fewer qualify, report the shortfall and expand within budget instead of weakening the rule invisibly. Freeze the manifest before tuning advice; reserve roughly one third for evaluation and split by repository where feasible. Neither policy tuning nor advisor context can access reserved-task oracle artifacts or validation outcomes.

Prefer the 111-task July pool, starting with Python and inexpensive Go/TypeScript tasks and expanding by measured environment cost. Use the older 110-task `2026_03` Hugging Face split if the Harbor path blocks progress. Record task dates relative to documented model cutoffs and avoid claims of complete decontamination where cutoff/training exposure is unknown.

## Advisor experiment after selection

Compare small-only, small-plus-advisor, and Sol-only on the frozen suite with repeated independent runs. All arms share the same task environments and action limits. Give the hybrid a lower dollar cap and report its actual spend rather than assuming savings from per-token prices.

Begin with a simple controller: actor-raised requests plus repeated visible test failures or repeated unsuccessful edits, evidence readiness, cooldown, and enough remaining budget to verify advice. Cap consultations at 1–2 per task. Use compact packets containing the issue, relevant code/diff, attempted hypotheses, visible test output, a focused question, and remaining budget. Do not provide full frontier solutions or hidden verifier feedback.

Use checkpoint replays on the development tasks to compare advice against a no-advice continuation with the same starting workspace and remaining actor budget. Then evaluate the frozen policy on reserved tasks. Include a small-model continuation with the same added spending allowance to distinguish advice value from simply buying more inference.

Report resolved rate, gap closed `(hybrid - small) / (frontier - small)` when the denominator is positive, hybrid/frontier cost ratio, cost per resolved task, latency, and advisor use. A possible research target is closing at least half the gap at no more than 30–40% of frontier-only inference cost; this is a hypothesis, not a promised result. Report infrastructure cost separately and include discovery/tuning costs alongside recurring evaluation costs. This selected suite is a diagnostic set and does not estimate overall SWE-rebench performance.

## Budget planning

The earlier leaderboard window reports approximately $0.27/task for Qwen3.6 and $0.85/task for GPT-5.6 Sol, with substantial caching. Use these only as rough planning references. Actual GPT-6.1 costs, provider routing, reasoning consumption, and cache behavior must be measured during smoke runs. A $100 discovery cap is a reasonable proposed starting point, with screening and confirmation allocations adjusted after calibration. Budget confirmation is pending; no paid work starts from this estimate alone.

## Sources

- [SWE-rebench leaderboard and methodology](https://swe-rebench.com/)
- [SWE-rebench methodology](https://swe-rebench.com/about)
- [Leaderboard dataset and July Harbor pointer](https://huggingface.co/datasets/nebius/SWE-rebench-leaderboard)
- [July 2026 Harbor snapshot](https://hub.harborframework.com/datasets/ibragim-badertdinov/swe-rebench-07-2026/latest)
- [SWE-rebench evaluation fork](https://github.com/SWE-rebench/SWE-bench-fork)
- [SWE-rebench V2 evaluator](https://github.com/SWE-rebench/SWE-rebench-V2)
- [mini-SWE-agent model adapters](https://mini-swe-agent.com/v2/reference/models/overview/)
- [OpenRouter model catalog](https://openrouter.ai/api/v1/models)
- [Qwen3.6-35B-A3B providers](https://openrouter.ai/qwen/qwen3.6-35b-a3b)
- [GPT-6.1 Sol on OpenRouter](https://openrouter.ai/openai/gpt-6.1-sol)
- [GPT-6.1 Sol official model guidance](https://developers.openai.com/api/docs/models/gpt-6.1-sol)

