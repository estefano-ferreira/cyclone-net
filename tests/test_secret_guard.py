"""Tests for the security layer: canonical redaction and scanner patterns."""
import json

from security_layer.secret_guard import (
    REDACTION_MARKER,
    find_secret_in_line,
    redact_secrets,
)
from src.utils.snapshot import save_run_snapshot


def test_snapshot_never_contains_fake_secret(tmp_path):
    """A config carrying a fake credential must be redacted BEFORE the run
    snapshot reaches disk (spec item: the fake key string must not appear)."""
    cfg = {
        "download": {"cds_api": {"url": "https://example", "key": "FAKE_KG_20260712_XYZ"}},
        "tchp": {"copernicus": {"username": "fake_user_kg", "password": "FAKE_PW_777"}},
        "labels": {"ri_threshold_kt_24h": 30.0},
    }
    save_run_snapshot(cfg, tmp_path, "test")
    text = (tmp_path / "run_snapshot.json").read_text(encoding="utf-8")

    assert "FAKE_KG_20260712_XYZ" not in text
    assert "FAKE_PW_777" not in text
    assert "fake_user_kg" not in text
    assert REDACTION_MARKER in text
    # Non-sensitive values survive.
    assert json.loads(text)["config"]["labels"]["ri_threshold_kt_24h"] == 30.0


def test_redact_secrets_is_canonical_in_security_layer():
    """src.utils.redaction must re-export the security layer implementation."""
    from security_layer import secret_guard
    from src.utils import redaction

    assert redaction.redact_secrets is secret_guard.redact_secrets


def test_scanner_flags_uuid_shaped_cds_key():
    # SYNTHETIC uuid — never place a real (even rotated) credential here.
    assert find_secret_in_line("key: deadbeef-0000-4000-8000-feedfacecafe")


def test_scanner_flags_password_assignment():
    # Quoted values may contain '#'/'@' (the leaked password's shape did);
    # these values are SYNTHETIC.
    assert find_secret_in_line('password: "Fk@#98765Pw"')
    assert find_secret_in_line("token = 'abc123DEF456'")


def test_scanner_ignores_code_identifiers_and_placeholders():
    # Regression: cache-dict assignment must not be flagged (no digits in value).
    assert find_secret_in_line("key = str(path)") is None
    assert find_secret_in_line('key: ""') is None
    assert find_secret_in_line("password: ***REDACTED***") is None
