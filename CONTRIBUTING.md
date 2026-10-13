# Contributing

Use Python 3.12 or newer on macOS or Linux and the checked-in dependency lock:

```sh
uv sync --locked --extra runtime
MSWEA_SILENT_STARTUP=1 LITELLM_LOCAL_MODEL_COST_MAP=True \
  uv run --locked --extra runtime python scripts/check_tests.py --require-runtime
```

The suite performs offline unit/runtime integration tests using scripted model transports and temporary fixture repositories. It requires no API key, Docker or downloaded benchmark. The runtime mode fails if tests are unexpectedly skipped. Run the lightweight suite in a separate environment without the runtime extra to check optional-dependency boundaries.

After a major change, review the diff, run the relevant behavior checks, then run the full offline suite before proceeding. New tests should verify meaningful contracts: isolation, billing uncertainty, resumption, policy dispatch, advice evidence or installed-package usability. Preserve existing work and keep changes scoped.

Build and verify the installable artifact:

```sh
uv build
uv venv '/tmp/autoresearch wheel check'
uv pip install --python '/tmp/autoresearch wheel check/bin/python' \
  'dist/autoresearch_budget-0.1.0-py3-none-any.whl[runtime]'
MSWEA_SILENT_STARTUP=1 LITELLM_LOCAL_MODEL_COST_MAP=True \
  '/tmp/autoresearch wheel check/bin/python' scripts/verify_install.py
```

`verify_install.py` operates outside the checkout in paths containing spaces, checks packaged presets, initializes a workspace, runs offline doctor/planning, completes the current advisory demo, and exports a report with working evidence links. CI also runs these checks on Linux and macOS. CI never dispatches paid inference.

Changes to prompts, action schemas, advice obligations, recovery evidence, limits or accounting alter the experiment contract. Document them in the changelog, preserve evidence/provenance, and review comparisons for confounding changes. Keep sandbox dependency coverage current when adding imports to uploaded modules.

Generated results, credentials, raw datasets, ledgers, benchmark patches, chart builds and historical analyses must remain outside tracked source. Use ignored `runs/`/`data/` locations or an external experiment archive. Review wheel/source-distribution contents as well as the Git diff before release. The internal publication roadmap is intentionally excluded from Git and distribution artifacts.

Live validation is a separate manual release check: choose a small pinned task, explicit routes and call limits, and an authorized spending allowance; run standalone and advisory attempts from the installed package and preserve their evidence externally. Do not enable live inference in CI.
