"""Build the UI-Tool-only source archive from an explicit allowlist.

The archive contains the controller, workers, launcher, model-preparation helper,
configuration templates and service documentation.  It deliberately omits the
OS-Critic evaluator, downloaded data, model weights, runtime output and local
configuration overrides so the result can be published independently.
"""

import hashlib
import gzip
from pathlib import Path
import tarfile


ROOT = Path(__file__).resolve().parents[1]
TOP_FILES = [
    '.gitignore',
    'README.md',
    'README.zh-CN.md',
    'THIRD_PARTY_NOTICES.md',
]
FILES = [
    'configs/local.env.example',
    'environments/ui-tool.yml',
    'environments/requirements-ui-tool.txt',
    # Optional VLM/benchmark environment recipes are small text files; keeping
    # them here makes the service documentation self-contained without shipping
    # evaluator code or any model/data artifacts.
    'environments/vllm.yml',
    'environments/requirements-vllm.txt',
    'environments/benchmark.yml',
    'environments/requirements-benchmark.txt',
    'environments/requirements-test.txt',
    'assets/smoke_ui.png',
    'assets/ui_tool_architecture.svg',
    'assets/method/method_20260824.png',
    'assets/method/method_20260824.pdf',
    'docs/ENVIRONMENT.md',
    'docs/EVALUATION.md',
    'docs/REPRODUCIBILITY.md',
    'docs/SERVICES.md',
    'docs/VALIDATION.md',
    'provenance/observed_environments.json',
    'provenance/reference_assets.json',
    'provenance/source_manifest.json',
    'third_party/OS-Critic-Bench.DATASET_CARD.md',
    'third_party/OS-Oracle.README.md',
    'scripts/build_tool_release.py',
    'scripts/check_services.py',
    'scripts/prepare_caption_bundle.py',
    'scripts/serve_vlm.sh',
    'scripts/start_ui_tool.sh',
]


def selected_files():
    paths = [ROOT / name for name in TOP_FILES + FILES]
    paths.extend(path for path in (ROOT / 'UI_Tool').rglob('*') if path.is_file())
    selected = []
    for path in sorted(set(paths)):
        relative = path.relative_to(ROOT)
        if path.is_symlink() or any(part in {
            '__pycache__', '.ipynb_checkpoints', 'runtime', 'logs', 'temp_images',
        } for part in relative.parts):
            continue
        if path.name in {'local.env', 'api_key.txt', 'api_base.txt'}:
            continue
        if path.suffix in {'.pyc', '.log', '.safetensors', '.pt', '.pth'}:
            continue
        selected.append(path)
    return selected


def main():
    selected = selected_files()
    missing = [str(path.relative_to(ROOT)) for path in selected if not path.is_file()]
    if missing:
        raise FileNotFoundError(', '.join(missing))

    dist = ROOT / 'dist'
    dist.mkdir(exist_ok=True)
    archive = dist / 'GUI-PRA-ui-tool-source.tar.gz'
    # Stable tar and gzip metadata makes the checksum reproducible across builds.
    with archive.open('wb') as raw, gzip.GzipFile(fileobj=raw, mode='wb', mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode='w') as tar:
            for path in selected:
                relative = path.relative_to(ROOT)
                info = tar.gettarinfo(str(path), arcname=str(Path(ROOT.name) / relative))
                info.uid = info.gid = 0
                info.uname = info.gname = ''
                info.mtime = 0
                with path.open('rb') as stream:
                    tar.addfile(info, stream)

    checksum = hashlib.sha256(archive.read_bytes()).hexdigest()
    (dist / (archive.name + '.sha256')).write_text(f'{checksum}  {archive.name}\n')
    print(f'{archive}: {len(selected)} files, {archive.stat().st_size} bytes')


if __name__ == '__main__':
    main()
