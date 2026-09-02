from __future__ import annotations

import importlib.util
import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "download_annotation_generator.py"

MINIMAL_SNAPSHOT_FILES = {
    "LICENSE": b"Apache License 2.0\n",
    "README.md": b"# Qwen2.5-7B-Instruct\n",
    "config.json": b'{"model_type":"qwen2"}\n',
    "generation_config.json": b"{}\n",
    "tokenizer.json": b"{}\n",
    "tokenizer_config.json": b"{}\n",
    "model.safetensors": b"safe-placeholder",
}


def load_downloader():
    assert SCRIPT.is_file(), "production implementation is missing"
    spec = importlib.util.spec_from_file_location("download_annotation_generator", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_minimal_snapshot(directory: Path) -> None:
    for relative_path, payload in MINIMAL_SNAPSHOT_FILES.items():
        path = directory / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)


def metadata_for(files: dict[str, bytes]) -> dict[str, tuple[int, str]]:
    return {
        path: (len(payload), hashlib.sha256(payload).hexdigest())
        for path, payload in files.items()
    }


def use_minimal_trust_anchor(downloader) -> None:
    downloader.TRUSTED_FILES = metadata_for(MINIMAL_SNAPSHOT_FILES)


class DownloadAnnotationGeneratorTests(unittest.TestCase):
    def test_pins_official_repo_and_revision_and_enforces_allowlist(self) -> None:
        downloader = load_downloader()

        self.assertEqual(downloader.REPO_ID, "Qwen/Qwen2.5-7B-Instruct")
        self.assertEqual(downloader.REVISION, "fe11104b620d588ccc049ff6631dd3ea002e3d98")
        self.assertTrue(
            hasattr(downloader, "TRANSPORT_ENDPOINT"),
            "China mirror endpoint support is missing",
        )
        self.assertEqual(downloader.TRANSPORT_ENDPOINT, "https://modelscope.cn")
        self.assertEqual(
            downloader.TRANSPORT_REVISION,
            "16c174980d8a1492910551634b4969e69cdc2444",
        )
        self.assertTrue(downloader.is_allowed_file("model-00001-of-00004.safetensors"))
        self.assertTrue(downloader.is_allowed_file("model.safetensors.index.json"))
        self.assertTrue(downloader.is_allowed_file("tokenizer.json"))
        self.assertTrue(downloader.is_allowed_file("README.md"))
        self.assertTrue(downloader.is_allowed_file("LICENSE"))

        for unsafe in (
            "pytorch_model.bin",
            "weights.pkl",
            "modeling_qwen.py",
            "run.exe",
            "nested/config.json",
            "../config.json",
            "unknown.txt",
        ):
            with self.subTest(unsafe=unsafe):
                self.assertFalse(downloader.is_allowed_file(unsafe))


    def test_install_uses_sibling_staging_then_writes_manifest(self) -> None:
        downloader = load_downloader()
        use_minimal_trust_anchor(downloader)
        with tempfile.TemporaryDirectory() as temporary_directory:
            tmp_path = Path(temporary_directory)
            target = tmp_path / "Qwen2.5-7B-Instruct"
            observed: dict[str, Path] = {}

            def fetch(staging: Path) -> None:
                observed["staging"] = staging
                self.assertNotEqual(staging, target)
                self.assertEqual(staging.parent, target.parent)
                self.assertIn(".staging-", staging.name)
                write_minimal_snapshot(staging)

            manifest = downloader.install_model(target, fetcher=fetch)

            self.assertTrue(target.is_dir())
            self.assertFalse(observed["staging"].exists())
            self.assertEqual(manifest["repo"], downloader.REPO_ID)
            self.assertEqual(manifest["revision"], downloader.REVISION)
            self.assertEqual(len(manifest["revision"]), 40)
            self.assertTrue((target / "manifest.json").is_file())
            self.assertEqual(downloader.verify_directory(target)["revision"], downloader.REVISION)


    def test_manifest_records_sorted_relative_paths_sizes_and_hashes(self) -> None:
        downloader = load_downloader()
        use_minimal_trust_anchor(downloader)
        with tempfile.TemporaryDirectory() as temporary_directory:
            tmp_path = Path(temporary_directory)
            write_minimal_snapshot(tmp_path)

            manifest = downloader.write_manifest(tmp_path)
            entries = manifest["files"]

            self.assertEqual(
                [entry["path"] for entry in entries],
                sorted(entry["path"] for entry in entries),
            )
            self.assertTrue(all(set(entry) == {"path", "bytes", "sha256"} for entry in entries))
            self.assertTrue(all(entry["bytes"] > 0 for entry in entries))
            self.assertTrue(all(len(entry["sha256"]) == 64 for entry in entries))
            self.assertTrue(manifest["downloaded_at"].endswith("Z"))
            self.assertIn("transport_endpoint", manifest)
            self.assertEqual(manifest["transport_endpoint"], downloader.TRANSPORT_ENDPOINT)
            self.assertIn("transport_revision", manifest)
            self.assertEqual(manifest["transport_revision"], downloader.TRANSPORT_REVISION)

    def test_network_fetch_explicitly_uses_pinned_modelscope_transport(self) -> None:
        downloader = load_downloader()
        self.assertTrue(
            hasattr(downloader, "fetch_from_modelscope"),
            "ModelScope transport support is missing",
        )
        remote_files = {
            "LICENSE": b"license",
            "README.md": b"readme",
            "config.json": b'{"model_type":"qwen2"}',
            "generation_config.json": b"{}",
            "tokenizer.json": b"{}",
            "tokenizer_config.json": b"{}",
            "model.safetensors": b"weights",
        }
        downloader.TRUSTED_FILES = metadata_for(remote_files)
        calls: list[str] = []

        class FakeResponse:
            def __init__(self, *, payload=b"", body=None, url="https://modelscope.cn"):
                self.payload = payload
                self.body = body
                self.url = url
                self.history = []

            def raise_for_status(self):
                return None

            def json(self):
                return self.body

            def iter_content(self, chunk_size):
                yield self.payload

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        class FakeSession:
            def get(self, url, **kwargs):
                calls.append(url)
                if url.endswith("/repo/files"):
                    self_params = kwargs["params"]
                    self_outer.assertEqual(
                        self_params["Revision"], downloader.TRANSPORT_REVISION
                    )
                    return FakeResponse(
                        body={
                            "Code": 200,
                            "Data": {
                                "Files": [
                                    {
                                        "Path": name,
                                        "Size": len(payload),
                                        "Sha256": hashlib.sha256(payload).hexdigest(),
                                    }
                                    for name, payload in remote_files.items()
                                ]
                            },
                        },
                        url=url,
                    )
                filename = unquote(url.rsplit("/", 1)[-1])
                return FakeResponse(payload=remote_files[filename], url=url)

        self_outer = self

        with tempfile.TemporaryDirectory() as temporary_directory:
            downloader.fetch_from_modelscope(
                Path(temporary_directory), session=FakeSession()
            )

        self.assertEqual(len(calls), len(remote_files) + 1)
        self.assertTrue(all(url.startswith("https://modelscope.cn/") for url in calls))
        self.assertTrue(
            all(downloader.TRANSPORT_REVISION in url for url in calls[1:])
        )

    def test_transport_allows_modelscope_china_cdn_but_rejects_other_hosts(self) -> None:
        downloader = load_downloader()
        try:
            downloader._validate_transport_url(
                "https://cdn-lfs-cn-1.modelscope.cn/prod/lfs-objects/example"
            )
        except downloader.VerificationError as exc:
            self.fail(f"ModelScope China CDN should be allowed: {exc}")
        with self.assertRaises(downloader.VerificationError):
            downloader._validate_transport_url(
                "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct"
            )

    def test_redirect_is_validated_before_any_second_get(self) -> None:
        downloader = load_downloader()
        self.assertTrue(
            hasattr(downloader, "_safe_get"),
            "manual redirect handling is missing",
        )
        calls: list[tuple[str, dict[str, object]]] = []

        class RedirectResponse:
            status_code = 302
            headers = {"Location": "https://attacker.invalid/steal"}
            url = "https://modelscope.cn/redirect"

            def close(self):
                return None

        class FakeSession:
            def get(self, url, **kwargs):
                calls.append((url, kwargs))
                return RedirectResponse()

        with self.assertRaises(downloader.VerificationError):
            downloader._safe_get(
                FakeSession(),
                "https://modelscope.cn/start",
                timeout=(1, 1),
            )

        self.assertEqual(len(calls), 1, "external Location caused a second GET")
        self.assertIs(calls[0][1].get("allow_redirects"), False)

    def test_fetch_disables_environment_proxy_inheritance(self) -> None:
        downloader = load_downloader()

        class StopRequest(RuntimeError):
            pass

        class FakeSession:
            trust_env = True

            def get(self, url, **kwargs):
                raise StopRequest

        session = FakeSession()
        with self.assertRaises(StopRequest):
            downloader.fetch_from_modelscope(Path("unused"), session=session)
        self.assertFalse(session.trust_env)

    def test_code_pins_exact_13_file_trust_anchor(self) -> None:
        downloader = load_downloader()
        self.assertTrue(
            hasattr(downloader, "TRUSTED_FILES"),
            "exact artifact trust anchor is missing",
        )
        self.assertEqual(len(downloader.TRUSTED_FILES), 13)
        self.assertEqual(
            downloader.TRUSTED_FILES["generation_config.json"],
            (
                243,
                "3a8f9087e486054c8a4a08dae2e5a3ba62e23da212b5b8c08bc42cb983c3459f",
            ),
        )
        self.assertEqual(
            downloader.TRUSTED_FILES["tokenizer_config.json"],
            (
                7305,
                "5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583",
            ),
        )

    def test_remote_metadata_must_match_trusted_baseline_before_file_get(self) -> None:
        downloader = load_downloader()
        trusted_files = dict(MINIMAL_SNAPSHOT_FILES)
        remote_files = dict(trusted_files)
        remote_files["README.md"] = b"remotely changed readme"
        downloader.TRUSTED_FILES = metadata_for(trusted_files)
        calls: list[str] = []

        class FakeResponse:
            status_code = 200
            headers: dict[str, str] = {}
            history: list[object] = []

            def __init__(self, *, url, body=None, payload=b""):
                self.url = url
                self.body = body
                self.payload = payload

            def raise_for_status(self):
                return None

            def json(self):
                return self.body

            def iter_content(self, chunk_size):
                yield self.payload

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        class FakeSession:
            trust_env = True

            def get(self, url, **kwargs):
                calls.append(url)
                if url.endswith("/repo/files"):
                    return FakeResponse(
                        url=url,
                        body={
                            "Code": 200,
                            "Data": {
                                "Files": [
                                    {
                                        "Path": name,
                                        "Size": len(payload),
                                        "Sha256": hashlib.sha256(payload).hexdigest(),
                                    }
                                    for name, payload in remote_files.items()
                                ]
                            },
                        },
                    )
                return FakeResponse(url=url, payload=b"")

        with tempfile.TemporaryDirectory() as temporary_directory:
            with self.assertRaises(downloader.VerificationError):
                downloader.fetch_from_modelscope(
                    Path(temporary_directory), session=FakeSession()
                )
        self.assertEqual(len(calls), 1, "file GET occurred before baseline validation")

    def test_remanifested_local_tamper_fails_trusted_baseline(self) -> None:
        downloader = load_downloader()
        use_minimal_trust_anchor(downloader)
        with tempfile.TemporaryDirectory() as temporary_directory:
            tmp_path = Path(temporary_directory)
            write_minimal_snapshot(tmp_path)
            downloader.write_manifest(tmp_path)

            changed = b"tampered but remanifested\n"
            changed_path = tmp_path / "README.md"
            changed_path.write_bytes(changed)
            manifest_path = tmp_path / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            entry = next(item for item in manifest["files"] if item["path"] == "README.md")
            entry["bytes"] = len(changed)
            entry["sha256"] = hashlib.sha256(changed).hexdigest()
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaises(downloader.VerificationError):
                downloader.verify_directory(tmp_path)


    def test_verify_only_rejects_missing_tampered_or_extra_files(self) -> None:
        downloader = load_downloader()
        use_minimal_trust_anchor(downloader)
        for damage in ("missing", "tampered", "dangerous", "safe-extra", "endpoint"):
            with self.subTest(damage=damage), tempfile.TemporaryDirectory() as temporary_directory:
                tmp_path = Path(temporary_directory)
                write_minimal_snapshot(tmp_path)
                downloader.write_manifest(tmp_path)

                if damage == "missing":
                    (tmp_path / "config.json").unlink()
                elif damage == "tampered":
                    (tmp_path / "config.json").write_text('{"changed":true}\n', encoding="utf-8")
                elif damage == "dangerous":
                    (tmp_path / "payload.py").write_text("raise SystemExit\n", encoding="utf-8")
                elif damage == "safe-extra":
                    (tmp_path / "notes.txt").write_text("unmanifested\n", encoding="utf-8")
                else:
                    manifest_path = tmp_path / "manifest.json"
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    manifest["transport_endpoint"] = "https://huggingface.co"
                    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

                with self.assertRaises(downloader.VerificationError):
                    downloader.verify_directory(tmp_path)


    def test_failed_staging_validation_preserves_existing_target(self) -> None:
        downloader = load_downloader()
        use_minimal_trust_anchor(downloader)
        with tempfile.TemporaryDirectory() as temporary_directory:
            tmp_path = Path(temporary_directory)
            target = tmp_path / "Qwen2.5-7B-Instruct"
            target.mkdir()
            marker = target / "existing.marker"
            marker.write_text("keep me", encoding="utf-8")

            def unsafe_fetch(staging: Path) -> None:
                write_minimal_snapshot(staging)
                (staging / "pytorch_model.bin").write_bytes(b"pickle-like")

            with self.assertRaises(downloader.VerificationError):
                downloader.install_model(target, fetcher=unsafe_fetch)

            self.assertEqual(marker.read_text(encoding="utf-8"), "keep me")
            self.assertEqual(list(tmp_path.glob(".Qwen2.5-7B-Instruct.staging-*")), [])

    def test_second_promotion_rename_fails_with_durable_state_and_rolls_back(self) -> None:
        downloader = load_downloader()
        use_minimal_trust_anchor(downloader)
        with tempfile.TemporaryDirectory() as temporary_directory:
            tmp_path = Path(temporary_directory)
            target = tmp_path / "Qwen2.5-7B-Instruct"
            target.mkdir()
            write_minimal_snapshot(target)
            downloader.write_manifest(target)
            original_inode = target.stat().st_ino
            state_path = tmp_path / ".Qwen2.5-7B-Instruct.promotion.json"
            state_seen_before_second_rename = False
            real_rename = Path.rename

            def fail_second_rename(path: Path, destination: Path):
                nonlocal state_seen_before_second_rename
                if path.name.startswith(".Qwen2.5-7B-Instruct.staging-"):
                    state_seen_before_second_rename = state_path.is_file()
                    raise OSError("simulated second rename failure")
                return real_rename(path, destination)

            def fetch(staging: Path) -> None:
                write_minimal_snapshot(staging)

            with patch.object(Path, "rename", fail_second_rename):
                with self.assertRaisesRegex(OSError, "simulated second rename failure"):
                    downloader.install_model(target, fetcher=fetch)

            self.assertTrue(
                state_seen_before_second_rename,
                "promotion state was not durable before the second rename",
            )
            self.assertEqual(target.stat().st_ino, original_inode)
            self.assertFalse(state_path.exists())
            self.assertEqual(list(tmp_path.glob(".Qwen2.5-7B-Instruct.backup-*")), [])
            self.assertEqual(list(tmp_path.glob(".Qwen2.5-7B-Instruct.staging-*")), [])

    def test_startup_recovery_completes_verified_interrupted_promotion(self) -> None:
        downloader = load_downloader()
        use_minimal_trust_anchor(downloader)
        self.assertTrue(
            hasattr(downloader, "recover_promotion"),
            "durable promotion recovery is missing",
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            tmp_path = Path(temporary_directory)
            target = tmp_path / "Qwen2.5-7B-Instruct"
            staging = tmp_path / ".Qwen2.5-7B-Instruct.staging-interrupted"
            backup = tmp_path / ".Qwen2.5-7B-Instruct.backup-interrupted"
            for directory in (target, staging):
                directory.mkdir()
                write_minimal_snapshot(directory)
                downloader.write_manifest(directory)
            staging_inode = staging.stat().st_ino

            downloader._write_promotion_state(target, staging, backup)
            target.rename(backup)
            downloader.recover_promotion(target)

            self.assertTrue(target.is_dir())
            self.assertEqual(target.stat().st_ino, staging_inode)
            self.assertFalse(staging.exists())
            self.assertFalse(backup.exists())
            self.assertFalse(
                (tmp_path / ".Qwen2.5-7B-Instruct.promotion.json").exists()
            )

    def test_startup_recovery_restores_backup_when_staging_was_lost(self) -> None:
        downloader = load_downloader()
        use_minimal_trust_anchor(downloader)
        self.assertTrue(
            hasattr(downloader, "recover_promotion"),
            "durable promotion recovery is missing",
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            tmp_path = Path(temporary_directory)
            target = tmp_path / "Qwen2.5-7B-Instruct"
            staging = tmp_path / ".Qwen2.5-7B-Instruct.staging-lost"
            backup = tmp_path / ".Qwen2.5-7B-Instruct.backup-interrupted"
            target.mkdir()
            write_minimal_snapshot(target)
            downloader.write_manifest(target)
            original_inode = target.stat().st_ino

            downloader._write_promotion_state(target, staging, backup)
            target.rename(backup)
            downloader.recover_promotion(target)

            self.assertTrue(target.is_dir())
            self.assertEqual(target.stat().st_ino, original_inode)
            self.assertFalse(backup.exists())
            self.assertFalse(
                (tmp_path / ".Qwen2.5-7B-Instruct.promotion.json").exists()
            )


    def test_verify_only_cli_is_offline_and_returns_nonzero_for_damage(self) -> None:
        downloader = load_downloader()
        use_minimal_trust_anchor(downloader)
        with tempfile.TemporaryDirectory() as temporary_directory:
            tmp_path = Path(temporary_directory)
            write_minimal_snapshot(tmp_path)
            downloader.write_manifest(tmp_path)

            output = io.StringIO()
            error = io.StringIO()
            with redirect_stdout(output), redirect_stderr(error):
                return_code = downloader.main(
                    ["--verify-only", "--target-dir", str(tmp_path)]
                )
            self.assertEqual(return_code, 0, error.getvalue())
            self.assertIn("VERIFIED", output.getvalue())

            (tmp_path / "payload.py").write_text("print('unsafe')\n", encoding="utf-8")
            output = io.StringIO()
            error = io.StringIO()
            with redirect_stdout(output), redirect_stderr(error):
                return_code = downloader.main(
                    ["--verify-only", "--target-dir", str(tmp_path)]
                )
            self.assertNotEqual(return_code, 0)
            self.assertIn("verification failed", error.getvalue().lower())


if __name__ == "__main__":
    unittest.main()
