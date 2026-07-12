"""Tests for snapshot redaction of sensitive configuration values."""
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.redaction import redact_secrets
from src.utils.snapshot import save_run_snapshot


class TestRedactSecrets:
    """Test the redact_secrets function with various key types."""

    def test_redacts_lowercase_sensitive_keys(self):
        """Redact standard lowercase sensitive keys."""
        obj = {
            "key": "secret_value",
            "password": "pass123",
            "token": "tok456",
        }
        result = redact_secrets(obj)
        assert result["key"] == "***REDACTED***"
        assert result["password"] == "***REDACTED***"
        assert result["token"] == "***REDACTED***"

    def test_redacts_case_insensitively(self):
        """Key matching should be case-insensitive."""
        obj = {
            "PASSWORD": "pass123",
            "Api_Key": "key456",
            "SECRET": "secret789",
        }
        result = redact_secrets(obj)
        assert result["PASSWORD"] == "***REDACTED***"
        assert result["Api_Key"] == "***REDACTED***"
        assert result["SECRET"] == "***REDACTED***"

    def test_preserves_non_sensitive_values(self):
        """Non-sensitive keys should retain their values."""
        obj = {
            "name": "test",
            "version": "2.0.0",
            "path": "/data/models",
        }
        result = redact_secrets(obj)
        assert result["name"] == "test"
        assert result["version"] == "2.0.0"
        assert result["path"] == "/data/models"

    def test_redacts_nested_dicts(self):
        """Redaction should work recursively on nested dicts."""
        obj = {
            "download": {
                "tchp": {
                    "key": "FAKE_KEY_123",
                    "username": "user123",
                },
                "other": "value",
            }
        }
        result = redact_secrets(obj)
        assert result["download"]["tchp"]["key"] == "***REDACTED***"
        assert result["download"]["tchp"]["username"] == "***REDACTED***"
        assert result["download"]["other"] == "value"

    def test_redacts_values_in_lists(self):
        """Redaction should work with dicts inside lists."""
        obj = {
            "configs": [
                {"key": "secret1", "name": "cfg1"},
                {"key": "secret2", "name": "cfg2"},
            ]
        }
        result = redact_secrets(obj)
        assert result["configs"][0]["key"] == "***REDACTED***"
        assert result["configs"][1]["key"] == "***REDACTED***"
        assert result["configs"][0]["name"] == "cfg1"
        assert result["configs"][1]["name"] == "cfg2"

    def test_does_not_mutate_input(self):
        """The original object should not be modified."""
        obj = {
            "key": "secret_value",
            "name": "test",
        }
        original_key = obj["key"]
        result = redact_secrets(obj)
        assert obj["key"] == original_key
        assert result["key"] == "***REDACTED***"


class TestSaveRunSnapshot:
    """Test save_run_snapshot redaction integration."""

    def test_snapshot_redacts_secrets_in_file(self, tmp_path):
        """Secrets in config should be redacted in the saved snapshot."""
        cfg = {
            "download": {
                "tchp": {
                    "key": "FAKE_SECRET_XYZ",
                }
            },
            "model": "resnet",
        }
        save_run_snapshot(cfg, tmp_path, "test")

        snapshot_path = tmp_path / "run_snapshot.json"
        assert snapshot_path.exists()

        with open(snapshot_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Verify redaction marker is present
        assert "***REDACTED***" in content

        # Verify the fake secret is NOT in the file
        assert "FAKE_SECRET_XYZ" not in content

        # Verify non-sensitive values are preserved
        assert "resnet" in content

    def test_snapshot_preserves_structure(self, tmp_path):
        """Snapshot should maintain config structure with redacted values."""
        cfg = {
            "download": {
                "tchp": {
                    "key": "FAKE_SECRET_XYZ",
                }
            },
            "model": "resnet",
        }
        save_run_snapshot(cfg, tmp_path, "test")

        snapshot_path = tmp_path / "run_snapshot.json"
        with open(snapshot_path, "r", encoding="utf-8") as f:
            snapshot = json.load(f)

        # Check structure is intact
        assert "config" in snapshot
        assert "download" in snapshot["config"]
        assert "tchp" in snapshot["config"]["download"]
        assert "key" in snapshot["config"]["download"]["tchp"]

        # Check the key value is redacted
        assert snapshot["config"]["download"]["tchp"]["key"] == "***REDACTED***"

        # Check non-sensitive values
        assert snapshot["config"]["model"] == "resnet"
