# Autoresearch on a budget

Small models drive a software experimentation loop and consult a frontier model for bounded advice. The first milestone is a reproducible SWE-rebench subset of 10–20 tasks with a measured small/frontier performance gap.

## Status

Initial research started on October 3, 2026. No paid model calls have been made. Harbor 0.23.0 has now been smoke-tested on one Python task and one Go task: both unmodified checkouts scored 0 and both packaged oracle solutions scored 1. The experiment runner and empirically validated model-gap subset are not implemented yet. The presentation's implementation and illustrative results are unverified inputs, not a baseline or architecture constraint for this project.

The local `.env` uses `OPENROUTER_KEY`; the eventual runner must accept it and supply the `OPENROUTER_API_KEY` expected by the agent library without logging either value or passing credentials to task containers.

## Findings

- The user selected `qwen/qwen3.6-35b-a3b` as the small model and `openai/gpt-5.6-sol` with medium reasoning as both the frontier baseline and advisor. Both models are invoked through OpenRouter. Qwen thinking settings and provider routing will be pinned after compatibility checks.
- The original Qwen3.8-35B-A3B request was clarified to Qwen3.6-35B-A3B. GPT-6.1 Sol was superseded by the user's selection of GPT-5.6 Sol. Official GPT-5.6 Sol pricing is $4/M input and $20/M output before caching and other adjustments; measure the actual chosen OpenRouter provider's billed cost.
- SWE-rebench's Hugging Face leaderboard dataset has 860 unique tasks in `test`, duplicated across named monthly splits. Do not concatenate those splits with `test` as independent tasks. The latest July 2026 evaluation distribution is in Harbor instead.
- Harbor's July snapshot advertises 111 hard tasks across Go, Java, Python, Rust, and TypeScript, version 1.0.0, revision 1. It is a snapshot label, not a guarantee that tasks were created in July.
- The public leaderboard's May 15–July 1, 2026 window reports Qwen3.6-35B-A3B at 24.7% mean resolved rate and GPT-5.6 Sol medium at 62.3%. These now correspond to the selected model names, but remain aggregate results under the leaderboard's setup rather than verified per-task results under our runner. Python alone has a 14% vs 48% gap in this window, but only 20 tasks; a Python-only pool may not yield enough stable differential tasks.
- The inspected page payload exposes task identities and aggregate statistics, not individual task verdicts. No reusable exact-pair outcome matrix has been established.
- Harbor supports local `docker` as its default sandbox on macOS when Docker Desktop or another compatible Docker runtime is installed. Harbor does not provide a special macOS task environment or verifier; it launches the task's container through the selected sandbox backend.
- The SWE-rebench and SWE-rebench V2 images inspected on Docker Hub are `linux/amd64`, not `linux/arm64` multi-architecture images. This Apple Silicon Mac can attempt them through Docker's x86 emulation, but compatibility and runtime cost must be measured. Do not assume local Mac execution is suitable for the experiment.
- Harbor also supports remote sandbox backends, including Modal, EC2, Daytona, and others. An x86 Linux backend is an execution option if local emulation is slow or unreliable; Daytona has no privileged status in this project.
- mini-SWE-agent is one candidate scaffold because it is small and supports OpenRouter. It has not been selected. Compare it with a minimal project-owned loop and Harbor's relevant pre-integrated agents using the same tasks, budgets, and model settings before freezing the harness.
- The downloaded July package contains 111 tasks across 65 repositories and occupies 11 MB before images. A generated manifest at `research/july_2026_manifest.json` contains identities, public metadata, resource requests, and verifier shape without copying oracle or test patch bodies.
- Zero-inference validation passed on `teng-lin__notebooklm-py-826` (Python/pytest) and `gotd__td-1759` (Go/go test). In each case, the base image matched the declared base commit, contained neither `/tests` nor `/solution`, the no-op run scored 0, and the oracle run scored 1.
- Docker Desktop's credential helper hung on anonymous pulls on this host. `scripts/with_anonymous_docker.sh` works around it with an ephemeral credential-free Docker config while retaining Docker Compose. It does not modify the user's persistent Docker configuration.
- Both tested images are `linux/amd64`; execution reported `x86_64` inside the containers, confirming emulation. Cached environment setup plus verification took roughly 8–14 seconds for the selected tasks, but downloading the two images reduced host free space from 66 GiB to 59 GiB. Local serial execution is viable for cheap tasks, while broader screening still needs image-size and resource filtering.

## Proposed implementation

Treat the benchmark evaluator and the agent scaffold as separate choices. First validate Harbor's July task packaging and verifier against the source SWE-rebench data and gold behavior. Independently compare candidate agent loops using a few tasks. Select the simplest scaffold that gives both models the same effective tools, preserves complete trajectories and costs, and reproduces expected grading. A small project-owned loop is acceptable and may be preferable if an existing agent adds hidden retries, summarization, model-specific behavior, or difficult-to-control token usage.

Harbor is a benchmark execution framework, not a required agent. A task environment contains the pinned repository and dependencies, and its verifier independently runs tests and grades the submitted change. Whichever agent loop we select must drive both standalone models under the same interface. The frontier advisor only returns advice.

Docker is not needed to call OpenRouter; inference happens at the provider. Containers supply disposable, reproducible environments for repository code and tests. On this Mac, run one zero-inference verifier/gold smoke test and one short agent task before relying on local Docker. Record wall time, peak memory, image-pull size, architecture warnings, test result, and any emulation failures. If it is unsuitable, move task execution to an x86 Linux sandbox while keeping inference and the experimental design unchanged.

Pin the agent version, task distribution revision, repository commit, prompts, reasoning settings, provider routing, and image digest. Record patches, trajectories, token usage including reasoning and cached tokens, actual billed cost, test outcomes, elapsed time, infrastructure errors, and advisor calls. Use resumable task/run IDs and reserve worst-case request cost before issuing a call so concurrent runs cannot overspend.

Expose a common action contract to both models and verify the OpenRouter adapter's compatibility with the selected pair. Shell-only, native tool calls, and any existing scaffold are implementation candidates to measure, rather than assumptions inherited from the talk. The advisor gets a compact evidence packet and returns diagnosis or a suggested experiment; the small actor owns edits and execution.

Agent containers receive the issue, base repository, and public tests. Oracle patches, benchmark test patches, verifier scripts, solution-bearing repository history, and benchmark verdicts stay outside the actor/advisor context. Audit prebuilt images for these artifacts. Use final independent grading for selection and reporting, while controller triggers use only actor-visible evidence. Verify both that the unmodified repository fails required issue tests and that the gold solution passes with regression tests preserved.

## Proposed discovery sequence

1. Validate the evaluation layer without model inference: inspect the July manifest, audit task/verifier isolation, run the unmodified and gold patches on 1–2 tasks, and measure local Mac Docker compatibility. Compare the result against the source SWE-rebench evaluator where practical.
2. Compare candidate agent scaffolds on 2–3 tasks using short, capped runs. Select the scaffold based on controllability, parity across the two models, complete cost/trajectory capture, and successful grading. Then test one normal request per model and 2–3 complete task runs before scaling.
3. Inspect the full candidate manifest, filter for reproducible CPU environments and clear instructions, and measure evaluator runtime with gold checks. Favor cheap tests and diverse repositories. Avoid selecting only tiny gold patches as a proxy for difficulty. Keep a manifest of all exclusions and deduplicate by task ID/base commit.
4. Screen 20 candidates first. Run the small model once. Run Sol on valid small-model failures first, then expand in batches of 10–20 as the budget permits. Infrastructure failures do not count as model failures. This sequence reduces frontier spending during discovery.
5. Repeat both standalone baselines twice more on promising differential tasks, from clean workspaces and under the same prompts, tools, and step/time limits. A provisional eligibility rule is small model 0/3 solved and Sol at least 2/3 solved. Three runs provide a screening estimate, not a precise probability. Retain failure categories and cost distributions.
6. Freeze 10–20 eligible tasks, ideally 15–20, spanning several repositories and failure types. If fewer qualify, report the shortfall and expand within budget instead of weakening the rule invisibly. Freeze the manifest before tuning advice; reserve roughly one third for evaluation and split by repository where feasible. Neither policy tuning nor advisor context can access reserved-task oracle artifacts or validation outcomes.

Prefer the 111-task July pool, starting with Python and inexpensive Go/TypeScript tasks and expanding by measured environment cost. Use the older 110-task `2026_03` Hugging Face split if the Harbor path blocks progress. Record task dates relative to documented model cutoffs and avoid claims of complete decontamination where cutoff/training exposure is unknown.

## Advisor experiment after selection

Compare small-only, small-plus-advisor, and Sol-only on the frozen suite with repeated independent runs. All arms share the same task environments and action limits. Give the hybrid a lower dollar cap and report its actual spend rather than assuming savings from per-token prices.

Begin with a simple controller: actor-raised requests plus repeated visible test failures or repeated unsuccessful edits, evidence readiness, cooldown, and enough remaining budget to verify advice. Cap consultations at 1–2 per task. Use compact packets containing the issue, relevant code/diff, attempted hypotheses, visible test output, a focused question, and remaining budget. Do not provide full frontier solutions or hidden verifier feedback.

Use checkpoint replays on the development tasks to compare advice against a no-advice continuation with the same starting workspace and remaining actor budget. Then evaluate the frozen policy on reserved tasks. Include a small-model continuation with the same added spending allowance to distinguish advice value from simply buying more inference.

Report resolved rate, gap closed `(hybrid - small) / (frontier - small)` when the denominator is positive, hybrid/frontier cost ratio, cost per resolved task, latency, and advisor use. A possible research target is closing at least half the gap at no more than 30–40% of frontier-only inference cost; this is a hypothesis, not a promised result. Report infrastructure cost separately and include discovery/tuning costs alongside recurring evaluation costs. This selected suite is a diagnostic set and does not estimate overall SWE-rebench performance.

## Budget planning

The leaderboard window reports approximately $0.27/task for Qwen3.6 and $0.85/task for GPT-5.6 Sol, with substantial caching. Use these only as rough planning references. Actual provider costs, routing, reasoning consumption, and cache behavior must be measured during smoke runs. A $100 discovery cap is a reasonable proposed starting point, with screening and confirmation allocations adjusted after calibration. Budget confirmation is pending; no paid work starts from this estimate alone.

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
- [GPT-5.6 Sol on OpenRouter](https://openrouter.ai/openai/gpt-5.6-sol)
- [GPT-5.6 Sol official model guidance](https://developers.openai.com/api/docs/models/gpt-5.6-sol)
- [Harbor local and remote sandboxes](https://docs.harborframework.com/sandboxes/pre-integrated-sandboxes)
- [Harbor verifier contract](https://docs.harborframework.com/tasks/verifier)
- [SWE-rebench V2 Docker images](https://hub.docker.com/u/swerebenchv2)

