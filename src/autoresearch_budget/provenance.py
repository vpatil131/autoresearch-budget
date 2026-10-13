"""Content identities for the runtime and selected task; no grading content."""
from __future__ import annotations

import hashlib
from importlib.metadata import PackageNotFoundError, distribution, version
import json
from pathlib import Path
import platform


def package_hash(directory: Path) -> str:
    hashes = {}
    for path in sorted(directory.rglob('*')):
        if path.is_symlink():
            raise ValueError('task packages containing symlinks cannot be fingerprinted safely')
        if path.is_file():
            hashes[str(path.relative_to(directory))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()


def runtime_identity(source_root: Path | None = None) -> dict:
    package = source_root / 'autoresearch_budget' if source_root else Path(__file__).resolve().parent
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(package.glob('*.py'))}
    if not hashes:
        raise ValueError('runtime source snapshot is empty')
    dependencies = {}
    for name in ('autoresearch-budget', 'harbor', 'mini-swe-agent'):
        try:
            dependencies[name] = version(name)
        except PackageNotFoundError:
            dependencies[name] = None
    # Capture the runtime dependency closure, excluding unrelated development
    # tools. Optional dependency markers are evaluated for this environment.
    try:
        from packaging.requirements import Requirement
    except ImportError:
        Requirement = None
    if Requirement:
        pending = ['harbor', 'mini-swe-agent']
        visited = set()
        while pending:
            name = pending.pop().lower().replace('_', '-')
            if name in visited:
                continue
            visited.add(name)
            try:
                installed = distribution(name)
            except PackageNotFoundError:
                dependencies[name] = None
                continue
            dependencies[name] = installed.version
            for requirement in installed.requires or []:
                parsed = Requirement(requirement)
                if parsed.marker is None or parsed.marker.evaluate({'extra': ''}):
                    pending.append(parsed.name)
    return {'source_sha256': hashes, 'python': platform.python_version(),
            'platform': platform.system(), 'dependencies': dependencies}


def trial_provenance(config, task_dir: Path, source_root: Path | None = None) -> dict:
    from .contracts import CONTRACT_VERSION, PHASE2_CONTRACT_VERSION, PHASE3_CONTRACT_VERSION, ADVICE_PROTOCOL_VERSION, RECOVERY_CONTRACT_VERSION
    from .advisor import AdvicePolicy
    manifest = config.consultation.get('task_manifest')
    data = json.loads((config.project_root / manifest).read_text()) if manifest else None
    return {'schema_version': 1, 'task_package_sha256': package_hash(task_dir),
            'task_manifest': data, 'runtime': runtime_identity(source_root),
            'contract_version': (RECOVERY_CONTRACT_VERSION if AdvicePolicy.load(config.consultation).recovery_enabled
                                 else PHASE3_CONTRACT_VERSION if AdvicePolicy.load(config.consultation).phase3
                                 else PHASE2_CONTRACT_VERSION if AdvicePolicy.load(config.consultation).policy == "advice_requested"
                                 else CONTRACT_VERSION) if config.agent.get('scaffold') == 'mini-swe-advisor' else 'mini-swe-2.4.6',
            'advice_protocol_version': ADVICE_PROTOCOL_VERSION if config.agent.get('scaffold') == 'mini-swe-advisor' else None}
