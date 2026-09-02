"""Offline tests for the pinned detailed-action BGE encoder downloader."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tools import download_detailed_action_encoder as downloader


REVISION = "0123456789abcdef0123456789abcdef01234567"
ENDPOINT = "https://example.invalid"


class FakeHfApi:
    instances: list["FakeHfApi"] = []
    sha = REVISION

    def __init__(self, *, endpoint: str) -> None:
        self.endpoint = endpoint
        self.requested_repo: str | None = None
        self.__class__.instances.append(self)

    def model_info(self, repo_id: str) -> SimpleNamespace:
        self.requested_repo = repo_id
        return SimpleNamespace(sha=self.sha)


def write_complete_encoder(root: Path, *, include_bin: bool = False) -> None:
    (root / "model.safetensors").write_bytes(b"safe weights")
    (root / "config.json").write_text('{"model_type":"bert"}', encoding="utf-8")
    (root / "tokenizer.json").write_text('{"version":"1.0"}', encoding="utf-8")
    (root / "README.md").write_text("official model card", encoding="utf-8")
    if include_bin:
        (root / "pytorch_model.bin").write_bytes(b"unsafe pickle")


class DetailedActionEncoderDownloadTests(unittest.TestCase):
    def setUp(self) -> None:
        FakeHfApi.instances = []
        FakeHfApi.sha = REVISION

    def test_resolve_revision_requires_exactly_40_lowercase_hex_characters(self) -> None:
        with patch.object(downloader, "HfApi", FakeHfApi):
            self.assertEqual(downloader.resolve_revision("BAAI/model", ENDPOINT), REVISION)
        self.assertEqual(FakeHfApi.instances[0].endpoint, ENDPOINT)
        self.assertEqual(FakeHfApi.instances[0].requested_repo, "BAAI/model")

        for invalid in ("A" * 40, "a" * 39, "g" * 40, ""):
            FakeHfApi.sha = invalid
            with patch.object(downloader, "HfApi", FakeHfApi):
                with self.assertRaises(ValueError):
                    downloader.resolve_revision("BAAI/model", ENDPOINT)

    def test_download_passes_pinned_revision_endpoint_and_safe_allowlist(self) -> None:
        received: dict = {}

        def fake_snapshot_download(**kwargs: object) -> str:
            received.update(kwargs)
            write_complete_encoder(Path(str(kwargs["local_dir"])))
            return str(kwargs["local_dir"])

        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "encoder"
            with patch.object(downloader, "snapshot_download", fake_snapshot_download):
                manifest = downloader.download_encoder(
                    "BAAI/model", REVISION, destination, ENDPOINT
                )

            self.assertEqual(received["repo_id"], "BAAI/model")
            self.assertEqual(received["revision"], REVISION)
            self.assertEqual(received["endpoint"], ENDPOINT)
            self.assertTrue(received["allow_patterns"])
            allowed = tuple(received["allow_patterns"])
            self.assertIn("*.safetensors", allowed)
            self.assertIn("config.json", allowed)
            self.assertIn("tokenizer.json", allowed)
            self.assertFalse(any("bin" in pattern.lower() for pattern in allowed))
            self.assertEqual(manifest["revision"], REVISION)
            self.assertTrue((destination / "manifest.json").is_file())

    def test_download_rejects_bin_artifact_before_replacing_destination(self) -> None:
        def fake_snapshot_download(**kwargs: object) -> str:
            write_complete_encoder(Path(str(kwargs["local_dir"])), include_bin=True)
            return str(kwargs["local_dir"])

        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "encoder"
            destination.mkdir()
            (destination / "sentinel.txt").write_text("keep me", encoding="utf-8")
            with patch.object(downloader, "snapshot_download", fake_snapshot_download):
                with self.assertRaises(ValueError):
                    downloader.download_encoder("BAAI/model", REVISION, destination, ENDPOINT)
            self.assertEqual((destination / "sentinel.txt").read_text(encoding="utf-8"), "keep me")
            self.assertFalse((destination / "pytorch_model.bin").exists())

    def test_download_rejects_non_license_file_disguised_with_a_license_prefix(self) -> None:
        def fake_snapshot_download(**kwargs: object) -> str:
            root = Path(str(kwargs["local_dir"]))
            write_complete_encoder(root)
            (root / "LICENSE.exe").write_bytes(b"not a license")
            return str(root)

        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "encoder"
            with patch.object(downloader, "snapshot_download", fake_snapshot_download):
                with self.assertRaises(ValueError):
                    downloader.download_encoder("BAAI/model", REVISION, destination, ENDPOINT)
            self.assertFalse(destination.exists())

    def test_download_writes_deterministic_manifest_hashes_and_sizes(self) -> None:
        def fake_snapshot_download(**kwargs: object) -> str:
            write_complete_encoder(Path(str(kwargs["local_dir"])))
            return str(kwargs["local_dir"])

        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "encoder"
            with patch.object(downloader, "snapshot_download", fake_snapshot_download):
                manifest = downloader.download_encoder("BAAI/model", REVISION, destination, ENDPOINT)

            files = manifest["files"]
            self.assertEqual([item["path"] for item in files], sorted(item["path"] for item in files))
            expected = {
                path.relative_to(destination).as_posix(): {
                    "bytes": len(path.read_bytes()),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
                for path in destination.rglob("*")
                if path.is_file() and path.name != "manifest.json"
            }
            self.assertEqual(
                files,
                [{"path": name, **expected[name]} for name in sorted(expected)],
            )
            self.assertEqual(manifest["manifest_format"], "detailed_action_encoder_manifest_v1")
            self.assertEqual(manifest["repo_id"], "BAAI/model")
            self.assertEqual(manifest["endpoint"], ENDPOINT)
            self.assertEqual(
                json.loads((destination / "manifest.json").read_text(encoding="utf-8")), manifest
            )

    def test_incomplete_staging_preserves_existing_destination(self) -> None:
        def fake_snapshot_download(**kwargs: object) -> str:
            root = Path(str(kwargs["local_dir"]))
            (root / "model.safetensors").write_bytes(b"safe weights")
            (root / "config.json").write_text("{}", encoding="utf-8")
            return str(root)

        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "encoder"
            destination.mkdir()
            (destination / "sentinel.txt").write_text("keep me", encoding="utf-8")
            with patch.object(downloader, "snapshot_download", fake_snapshot_download):
                with self.assertRaises(ValueError):
                    downloader.download_encoder("BAAI/model", REVISION, destination, ENDPOINT)
            self.assertEqual((destination / "sentinel.txt").read_text(encoding="utf-8"), "keep me")
            self.assertFalse((destination / "manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
