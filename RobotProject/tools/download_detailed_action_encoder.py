"""Download a pinned BGE encoder through an auditable, safe staging directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
import uuid
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


DEFAULT_REPO = "BAAI/bge-small-zh-v1.5"
DEFAULT_DESTINATION = Path("models/pretrained/BAAI_bge-small-zh-v1.5")
DEFAULT_ENDPOINT = "https://hf-mirror.com"
MANIFEST_FORMAT = "detailed_action_encoder_manifest_v1"
ALLOW_PATTERNS = (
    "*.safetensors",
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.txt",
    "merges.txt",
    "spiece.model",
    "sentencepiece.bpe.model",
    "modules.json",
    "*/config.json",
    "README.md",
    "LICENSE",
    "LICENSE.md",
    "LICENSE.txt",
)
_JSON_ARTIFACTS = {
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "modules.json",
}
_TOKENIZER_ARTIFACTS = {
    "tokenizer.json",
    "vocab.txt",
    "spiece.model",
    "sentencepiece.bpe.model",
}


def resolve_revision(repo_id: str, endpoint: str) -> str:
    """Return the repository's immutable, lowercase 40-character commit SHA."""
    revision = HfApi(endpoint=endpoint).model_info(repo_id).sha
    if not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError("model revision must be exactly 40 lowercase hexadecimal characters")
    return revision


def download_encoder(repo_id: str, revision: str, destination: Path, endpoint: str) -> dict:
    """Download, validate, manifest, and atomically install a pinned encoder."""
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError("model revision must be exactly 40 lowercase hexadecimal characters")

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent))
    backup: Path | None = None
    try:
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            endpoint=endpoint,
            allow_patterns=list(ALLOW_PATTERNS),
            local_dir=staging,
            local_dir_use_symlinks=False,
        )

        cache_directory = staging / ".cache"
        if cache_directory.exists():
            shutil.rmtree(cache_directory)

        files: list[dict[str, object]] = []
        names: set[str] = set()
        for artifact in sorted(staging.rglob("*"), key=lambda path: path.relative_to(staging).as_posix()):
            if artifact.is_dir():
                continue
            if artifact.is_symlink() or not artifact.is_file():
                raise ValueError(f"unsafe non-regular artifact: {artifact.relative_to(staging).as_posix()}")
            relative = artifact.relative_to(staging).as_posix()
            basename = artifact.name
            permitted = (
                basename.endswith(".safetensors")
                or basename in _JSON_ARTIFACTS
                or basename in {"vocab.txt", "merges.txt", "spiece.model", "sentencepiece.bpe.model"}
                or basename == "README.md"
                or basename in {"LICENSE", "LICENSE.md", "LICENSE.txt"}
            )
            if not permitted:
                raise ValueError(f"disallowed downloaded artifact: {relative}")
            payload = artifact.read_bytes()
            names.add(basename)
            files.append(
                {
                    "path": relative,
                    "bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            )

        if "model.safetensors" not in names:
            raise ValueError("staging is missing model.safetensors")
        if "config.json" not in names:
            raise ValueError("staging is missing config.json")
        if not names.intersection(_TOKENIZER_ARTIFACTS):
            raise ValueError("staging is missing a tokenizer artifact")

        manifest = {
            "manifest_format": MANIFEST_FORMAT,
            "repo_id": repo_id,
            "revision": revision,
            "endpoint": endpoint,
            "files": files,
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        if destination.exists():
            backup = destination.parent / f".{destination.name}.backup-{uuid.uuid4().hex}"
            os.replace(destination, backup)
        try:
            os.replace(staging, destination)
        except Exception:
            if backup is not None and not destination.exists():
                os.replace(backup, destination)
            raise
        if backup is not None:
            shutil.rmtree(backup)
            backup = None
        return manifest
    finally:
        if staging.exists():
            shutil.rmtree(staging)
        if backup is not None and backup.exists() and not destination.exists():
            os.replace(backup, destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--output", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--revision")
    args = parser.parse_args()
    pinned_revision = args.revision or resolve_revision(args.repo, args.endpoint)
    result = download_encoder(args.repo, pinned_revision, args.output, args.endpoint)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
