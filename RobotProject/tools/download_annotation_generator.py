"""Safely install the pinned offline model used to generate annotation candidates.

This module deliberately has no integration with the project's inference runtime.
It downloads only an explicit set of non-executable Hugging Face artifacts, verifies
them in a sibling staging directory, and promotes the staging directory only after
all integrity checks pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Callable
from urllib.parse import quote, urljoin, urlparse


REPO_ID = "Qwen/Qwen2.5-7B-Instruct"
REVISION = "fe11104b620d588ccc049ff6631dd3ea002e3d98"
TRANSPORT_ENDPOINT = "https://modelscope.cn"
TRANSPORT_REVISION = "16c174980d8a1492910551634b4969e69cdc2444"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGET = PROJECT_ROOT / "models" / "pretrained" / "Qwen2.5-7B-Instruct"
MAX_REDIRECTS = 5

# Approved artifact baseline for the pinned ModelScope source revision above.
# This code-owned trust anchor is intentionally independent of local manifest.json.
TRUSTED_FILES: dict[str, tuple[int, str]] = {
    "LICENSE": (
        11343,
        "832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e",
    ),
    "README.md": (
        6240,
        "f366f33bbf6bcadbb7d87f0a21a7b65584a56b8d58b0743c77c88bee625b93a6",
    ),
    "config.json": (
        663,
        "7463bb0ea78315365e6c6b74de4e73bbcc8359dfb0c5a737584e077d42c0b03c",
    ),
    "generation_config.json": (
        243,
        "3a8f9087e486054c8a4a08dae2e5a3ba62e23da212b5b8c08bc42cb983c3459f",
    ),
    "merges.txt": (
        1671839,
        "599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3",
    ),
    "model-00001-of-00004.safetensors": (
        3945441440,
        "a1333e6293854747c481288ea83b348226af178dd565c49b6f9495ba1966aba7",
    ),
    "model-00002-of-00004.safetensors": (
        3864726352,
        "f5d25a2772cb825164a2a2c0fb6d51a87e282abf21e4dd75bc5cfb3cd0ea6185",
    ),
    "model-00003-of-00004.safetensors": (
        3864726424,
        "8efdec4c1bc12317ae1a38dc42b595ce777738a64deea3fcb8a0a91381bcdfd5",
    ),
    "model-00004-of-00004.safetensors": (
        3556377672,
        "1a72d403cdf0c1ec3cb7f289f17b394a01e64394c2e9b3c0f94dbce3faf879bd",
    ),
    "model.safetensors.index.json": (
        27752,
        "624bf7c47cd12468fdc16e38a47cf4f19e0415b859a223ba3c027eed2f0e1028",
    ),
    "tokenizer.json": (
        7031645,
        "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539",
    ),
    "tokenizer_config.json": (
        7305,
        "5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583",
    ),
    "vocab.json": (
        2776833,
        "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
    ),
}

_ALLOWED_EXACT = frozenset(
    {
        "LICENSE",
        "README.md",
        "added_tokens.json",
        "config.json",
        "generation_config.json",
        "merges.txt",
        "model.safetensors.index.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
    }
)
_SAFETENSORS_NAME = re.compile(r"model(?:-\d{5}-of-\d{5})?\.safetensors\Z")
_REQUIRED_EXACT = frozenset(
    {
        "LICENSE",
        "README.md",
        "config.json",
        "generation_config.json",
        "tokenizer_config.json",
    }
)
_DANGEROUS_SUFFIXES = frozenset(
    {
        ".bin",
        ".ckpt",
        ".dll",
        ".dylib",
        ".exe",
        ".joblib",
        ".pickle",
        ".pkl",
        ".ps1",
        ".py",
        ".pyc",
        ".pyd",
        ".sh",
        ".so",
    }
)


class VerificationError(RuntimeError):
    """Raised when a snapshot does not satisfy the pinned integrity contract."""


def _promotion_state_path(target: Path) -> Path:
    return target.parent / f".{target.name}.promotion.json"


def _write_promotion_state(target: Path, staging: Path, backup: Path) -> Path:
    """Durably record a same-parent promotion before moving the live target."""

    target = Path(target).resolve()
    for path in (staging, backup):
        if Path(path).parent.resolve() != target.parent:
            raise VerificationError("promotion paths must be siblings of the target")
    state = {
        "protocol_version": 1,
        "target": target.name,
        "staging": Path(staging).name,
        "backup": Path(backup).name,
    }
    state_path = _promotion_state_path(target)
    temporary = state_path.with_name(f"{state_path.name}.tmp-{uuid.uuid4().hex}")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(state, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(state_path)
    return state_path


def _state_sibling(target: Path, name: object, prefix: str) -> Path:
    if not isinstance(name, str) or Path(name).name != name or not name.startswith(prefix):
        raise VerificationError(f"unsafe promotion state path: {name!r}")
    return target.parent / name


def _remove_tree(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def recover_promotion(target: Path) -> bool:
    """Recover or finish an interrupted durable directory promotion."""

    target = Path(target).resolve()
    state_path = _promotion_state_path(target)
    if not state_path.exists():
        return False
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise VerificationError(f"promotion state cannot be read: {exc}") from exc
    if state.get("protocol_version") != 1 or state.get("target") != target.name:
        raise VerificationError("promotion state does not match the requested target")
    staging = _state_sibling(
        target,
        state.get("staging"),
        f".{target.name}.staging-",
    )
    backup = _state_sibling(
        target,
        state.get("backup"),
        f".{target.name}.backup-",
    )

    if target.exists():
        try:
            verify_directory(target)
        except VerificationError:
            if not backup.exists():
                raise
            verify_directory(backup)
            failed = target.parent / f".{target.name}.failed-{uuid.uuid4().hex}"
            target.rename(failed)
            try:
                backup.rename(target)
            except BaseException:
                if failed.exists() and not target.exists():
                    failed.rename(target)
                raise
            _remove_tree(failed)
        _remove_tree(staging)
        _remove_tree(backup)
        state_path.unlink()
        return True

    if staging.exists():
        verify_directory(staging)
        staging.rename(target)
        verify_directory(target)
        _remove_tree(backup)
        state_path.unlink()
        return True

    if backup.exists():
        verify_directory(backup)
        backup.rename(target)
        state_path.unlink()
        return True

    raise VerificationError("promotion state has no recoverable target, staging, or backup")


def is_allowed_file(relative_path: str) -> bool:
    """Return whether a repository path is explicitly safe to download."""

    path = PurePosixPath(relative_path)
    if path.is_absolute() or len(path.parts) != 1 or path.name in {"", ".", ".."}:
        return False
    return path.name in _ALLOWED_EXACT or _SAFETENSORS_NAME.fullmatch(path.name) is not None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot_files(directory: Path, *, include_manifest: bool = False) -> list[Path]:
    files: list[Path] = []
    for path in directory.rglob("*"):
        if path.is_symlink():
            raise VerificationError(f"symbolic links are forbidden: {path.relative_to(directory)}")
        if not path.is_file():
            continue
        relative = path.relative_to(directory).as_posix()
        if relative == "manifest.json" and not include_manifest:
            continue
        files.append(path)
    return sorted(files, key=lambda item: item.relative_to(directory).as_posix())


def _validate_allowed_snapshot(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise VerificationError(f"snapshot directory does not exist: {directory}")

    files = _snapshot_files(directory)
    relative_names = {path.relative_to(directory).as_posix() for path in files}
    rejected = sorted(name for name in relative_names if not is_allowed_file(name))
    if rejected:
        dangerous = [name for name in rejected if Path(name).suffix.lower() in _DANGEROUS_SUFFIXES]
        label = "dangerous or unapproved" if dangerous else "unapproved"
        raise VerificationError(f"{label} snapshot files: {', '.join(rejected)}")

    trusted_names = set(TRUSTED_FILES)
    missing_trusted = sorted(trusted_names - relative_names)
    extra_trusted = sorted(relative_names - trusted_names)
    if missing_trusted:
        raise VerificationError(
            f"trusted snapshot files are missing: {', '.join(missing_trusted)}"
        )
    if extra_trusted:
        raise VerificationError(
            f"files outside the trusted snapshot are forbidden: {', '.join(extra_trusted)}"
        )

    missing = sorted(_REQUIRED_EXACT - relative_names)
    if missing:
        raise VerificationError(f"required snapshot files are missing: {', '.join(missing)}")
    if not any(_SAFETENSORS_NAME.fullmatch(name) for name in relative_names):
        raise VerificationError("no safetensors model weights found")
    if not ({"tokenizer.json", "vocab.json"} & relative_names):
        raise VerificationError("no tokenizer vocabulary artifact found")
    if any(path.stat().st_size <= 0 for path in files):
        raise VerificationError("snapshot contains an empty file")

    try:
        config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise VerificationError(f"invalid config.json: {exc}") from exc
    if config.get("model_type") != "qwen2":
        raise VerificationError("config.json does not identify a qwen2 model")

    index_path = directory / "model.safetensors.index.json"
    if index_path.exists():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
            referenced = set(index["weight_map"].values())
        except (OSError, UnicodeError, json.JSONDecodeError, KeyError, AttributeError) as exc:
            raise VerificationError(f"invalid safetensors index: {exc}") from exc
        invalid_references = sorted(name for name in referenced if not is_allowed_file(name))
        missing_references = sorted(name for name in referenced if name not in relative_names)
        if invalid_references:
            raise VerificationError(
                f"safetensors index contains unsafe paths: {', '.join(invalid_references)}"
            )
        if missing_references:
            raise VerificationError(
                f"safetensors index references missing shards: {', '.join(missing_references)}"
            )

    return files


def _trusted_file_entries(directory: Path) -> list[dict[str, object]]:
    files = _validate_allowed_snapshot(directory)
    entries: list[dict[str, object]] = []
    for path in files:
        relative = path.relative_to(directory).as_posix()
        expected_size, expected_hash = TRUSTED_FILES[relative]
        actual_size = path.stat().st_size
        if actual_size != expected_size:
            raise VerificationError(
                f"trusted size mismatch for {relative}: expected {expected_size}, got {actual_size}"
            )
        actual_hash = _sha256(path)
        if actual_hash != expected_hash:
            raise VerificationError(f"trusted SHA-256 mismatch for {relative}")
        entries.append(
            {"path": relative, "bytes": actual_size, "sha256": actual_hash}
        )
    return entries


def write_manifest(directory: Path) -> dict[str, object]:
    """Validate snapshot files, hash them, and atomically write manifest.json."""

    directory = Path(directory)
    entries = _trusted_file_entries(directory)
    manifest: dict[str, object] = {
        "manifest_version": 1,
        "repo": REPO_ID,
        "revision": REVISION,
        "transport_endpoint": TRANSPORT_ENDPOINT,
        "transport_revision": TRANSPORT_REVISION,
        "downloaded_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "files": entries,
    }
    temporary_manifest = directory / ".manifest.json.tmp"
    temporary_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_manifest.replace(directory / "manifest.json")
    return manifest


def verify_directory(directory: Path) -> dict[str, object]:
    """Completely offline verification of the manifest and every local file."""

    directory = Path(directory)
    manifest_path = directory / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise VerificationError(f"manifest.json cannot be read: {exc}") from exc

    if manifest.get("repo") != REPO_ID:
        raise VerificationError(f"manifest repo must be {REPO_ID}")
    if manifest.get("revision") != REVISION or len(str(manifest.get("revision", ""))) != 40:
        raise VerificationError(f"manifest revision must be pinned to {REVISION}")
    if manifest.get("transport_endpoint") != TRANSPORT_ENDPOINT:
        raise VerificationError(
            f"manifest transport endpoint must be pinned to {TRANSPORT_ENDPOINT}"
        )
    if manifest.get("transport_revision") != TRANSPORT_REVISION:
        raise VerificationError(
            f"manifest transport revision must be pinned to {TRANSPORT_REVISION}"
        )
    entries = manifest.get("files")
    if not isinstance(entries, list) or not entries:
        raise VerificationError("manifest files must be a non-empty list")

    manifest_paths: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"path", "bytes", "sha256"}:
            raise VerificationError("manifest contains a malformed file entry")
        relative = entry.get("path")
        if not isinstance(relative, str) or not is_allowed_file(relative):
            raise VerificationError(f"manifest contains an unsafe path: {relative!r}")
        if relative in manifest_paths:
            raise VerificationError(f"manifest contains a duplicate path: {relative}")
        manifest_paths.add(relative)
        expected_size, expected_hash = TRUSTED_FILES.get(relative, (None, None))
        if entry.get("bytes") != expected_size or entry.get("sha256") != expected_hash:
            raise VerificationError(f"manifest differs from trusted baseline for {relative}")

    trusted_paths = set(TRUSTED_FILES)
    if manifest_paths != trusted_paths:
        missing_trusted = sorted(trusted_paths - manifest_paths)
        extra_trusted = sorted(manifest_paths - trusted_paths)
        raise VerificationError(
            "manifest path set differs from trusted baseline: "
            f"missing={missing_trusted}, extra={extra_trusted}"
        )

    actual_files = _snapshot_files(directory, include_manifest=True)
    actual_paths = {path.relative_to(directory).as_posix() for path in actual_files}
    expected_paths = manifest_paths | {"manifest.json"}
    missing = sorted(expected_paths - actual_paths)
    extra = sorted(actual_paths - expected_paths)
    if missing:
        raise VerificationError(f"manifest files are missing: {', '.join(missing)}")
    if extra:
        raise VerificationError(f"unmanifested files are forbidden: {', '.join(extra)}")

    trusted_entries = _trusted_file_entries(directory)
    if entries != trusted_entries:
        raise VerificationError("manifest file entries differ from trusted local artifacts")

    return manifest


def _validate_transport_url(url: str) -> None:
    parsed = urlparse(url)
    hostname = parsed.hostname or ""
    if parsed.scheme != "https" or not (
        hostname == "modelscope.cn" or hostname.endswith(".modelscope.cn")
    ):
        raise VerificationError(f"transport escaped the approved China mirror: {url}")


def _safe_get(session, url: str, *, max_redirects: int = MAX_REDIRECTS, **kwargs):
    """Issue GET requests while validating every redirect before following it."""

    request_url = url
    request_kwargs = dict(kwargs)
    for redirect_count in range(max_redirects + 1):
        _validate_transport_url(request_url)
        response = session.get(
            request_url,
            allow_redirects=False,
            **request_kwargs,
        )
        status_code = getattr(response, "status_code", 200)
        if status_code not in {301, 302, 303, 307, 308}:
            _validate_transport_url(getattr(response, "url", request_url))
            return response
        if redirect_count >= max_redirects:
            close = getattr(response, "close", None)
            if close is not None:
                close()
            raise VerificationError(f"transport exceeded {max_redirects} redirects")
        location = getattr(response, "headers", {}).get("Location")
        if not location:
            close = getattr(response, "close", None)
            if close is not None:
                close()
            raise VerificationError("transport redirect is missing Location")
        next_url = urljoin(getattr(response, "url", request_url), location)
        _validate_transport_url(next_url)
        close = getattr(response, "close", None)
        if close is not None:
            close()
        request_url = next_url
        request_kwargs.pop("params", None)
    raise AssertionError("unreachable redirect state")


def fetch_from_modelscope(staging: Path, *, session=None) -> None:
    """Download allowlisted artifacts from a pinned ModelScope China snapshot."""

    try:
        import requests
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
    except ImportError as exc:
        raise RuntimeError("requests and urllib3 are required for downloading") from exc

    if session is None:
        session = requests.Session()
        retry = Retry(
            total=3,
            connect=3,
            read=3,
            backoff_factor=1.0,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset({"GET"}),
        )
        session.mount("https://", HTTPAdapter(max_retries=retry))
    session.trust_env = False

    listing_url = f"{TRANSPORT_ENDPOINT}/api/v1/models/{REPO_ID}/repo/files"
    _validate_transport_url(listing_url)
    response = _safe_get(
        session,
        listing_url,
        params={"Revision": TRANSPORT_REVISION, "Recursive": "true"},
        timeout=(30, 120),
    )
    response.raise_for_status()
    _validate_transport_url(response.url)
    try:
        body = response.json()
        if body.get("Code") != 200:
            raise VerificationError(f"ModelScope file listing failed: {body!r}")
        remote_files = body["Data"]["Files"]
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise VerificationError(f"invalid ModelScope file listing: {exc}") from exc

    selected_metadata: dict[str, tuple[int, str]] = {}
    for item in remote_files:
        try:
            name = item["Path"]
            size = item["Size"]
            sha256 = item["Sha256"]
        except (KeyError, TypeError) as exc:
            raise VerificationError(f"malformed ModelScope file metadata: {item!r}") from exc
        if not is_allowed_file(name):
            continue
        if name in selected_metadata:
            raise VerificationError(f"duplicate ModelScope file metadata: {name}")
        if not isinstance(size, int) or size <= 0:
            raise VerificationError(f"invalid ModelScope byte count for {name}")
        if not isinstance(sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", sha256):
            raise VerificationError(f"invalid ModelScope SHA-256 for {name}")
        selected_metadata[name] = (size, sha256)

    selected = sorted(selected_metadata)
    selected_set = set(selected)
    missing = sorted(_REQUIRED_EXACT - selected_set)
    if missing or not any(_SAFETENSORS_NAME.fullmatch(name) for name in selected):
        detail = f"missing required files: {', '.join(missing)}" if missing else "no safetensors"
        raise VerificationError(f"pinned remote snapshot is incomplete: {detail}")
    trusted_set = set(TRUSTED_FILES)
    if selected_set != trusted_set:
        raise VerificationError(
            "ModelScope artifact paths differ from trusted baseline: "
            f"missing={sorted(trusted_set - selected_set)}, "
            f"extra={sorted(selected_set - trusted_set)}"
        )
    for filename, metadata in selected_metadata.items():
        if metadata != TRUSTED_FILES[filename]:
            raise VerificationError(
                f"ModelScope metadata differs from trusted baseline for {filename}"
            )

    total_bytes = sum(size for size, _ in selected_metadata.values())
    completed_bytes = 0
    for completed_files, filename in enumerate(selected, start=1):
        expected_size, expected_hash = selected_metadata[filename]
        download_url = (
            f"{TRANSPORT_ENDPOINT}/models/{REPO_ID}/resolve/"
            f"{TRANSPORT_REVISION}/{quote(filename, safe='')}"
        )
        _validate_transport_url(download_url)
        temporary_path = staging / f".{filename}.part"
        digest = hashlib.sha256()
        byte_count = 0
        with _safe_get(
            session,
            download_url,
            stream=True,
            timeout=(30, 300),
        ) as download:
            download.raise_for_status()
            with temporary_path.open("wb") as stream:
                for chunk in download.iter_content(chunk_size=8 * 1024 * 1024):
                    if not chunk:
                        continue
                    stream.write(chunk)
                    digest.update(chunk)
                    byte_count += len(chunk)
        if byte_count != expected_size:
            raise VerificationError(
                f"ModelScope size mismatch for {filename}: expected {expected_size}, got {byte_count}"
            )
        if digest.hexdigest() != expected_hash:
            raise VerificationError(f"ModelScope SHA-256 mismatch for {filename}")
        temporary_path.replace(staging / filename)
        completed_bytes += byte_count
        print(
            f"DOWNLOADED {completed_files}/{len(selected)} files; "
            f"{completed_bytes / (1024 ** 3):.2f}/{total_bytes / (1024 ** 3):.2f} GiB"
        )


def install_model(
    target: Path = DEFAULT_TARGET,
    *,
    fetcher: Callable[[Path], None] = fetch_from_modelscope,
) -> dict[str, object]:
    """Download and verify in staging, then safely replace the formal directory."""

    target = Path(target).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    recover_promotion(target)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=str(target.parent))
    )
    backup: Path | None = None
    state_path = _promotion_state_path(target)
    promoted = False
    try:
        fetcher(staging)
        write_manifest(staging)
        manifest = verify_directory(staging)

        if target.exists():
            verify_directory(target)
            backup = target.parent / f".{target.name}.backup-{uuid.uuid4().hex}"
            _write_promotion_state(target, staging, backup)
            target.rename(backup)
        try:
            staging.rename(target)
            promoted = True
        except BaseException:
            if backup is not None and backup.exists() and not target.exists():
                backup.rename(target)
            if target.exists() and state_path.exists():
                state_path.unlink()
            raise
        if backup is not None:
            shutil.rmtree(backup)
            state_path.unlink()
        return manifest
    finally:
        if not promoted and staging.exists() and not state_path.exists():
            shutil.rmtree(staging)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Install or offline-verify the pinned Qwen annotation candidate model."
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=DEFAULT_TARGET,
        help=f"model directory (default: {DEFAULT_TARGET})",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="perform manifest and SHA-256 verification without network access",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        recover_promotion(args.target_dir)
        if args.verify_only:
            manifest = verify_directory(args.target_dir)
            print(
                f"VERIFIED {args.target_dir} at {manifest['revision']} "
                f"({len(manifest['files'])} files)"
            )
        else:
            manifest = install_model(args.target_dir)
            print(
                f"INSTALLED {args.target_dir} at {manifest['revision']} "
                f"({len(manifest['files'])} files)"
            )
        return 0
    except (OSError, RuntimeError, VerificationError) as exc:
        print(f"verification failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
