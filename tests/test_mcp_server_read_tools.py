"""Tests for mcp_server.py read-only tools (list_results, read_result)."""
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import mcp_server


@pytest.fixture
def results_dir(tmp_path, monkeypatch):
    """Create a temporary results directory with test files and monkeypatch _results_dir."""
    # Create metrics.json
    metrics = {"roc_auc": 0.61, "pr_auc": 0.44}
    (tmp_path / "metrics.json").write_text(json.dumps(metrics))

    # Create run_snapshot.json with sensitive values
    snapshot = {
        "config": {
            "download": {
                "tchp": {
                    "key": "SEGREDO_FAKE_123",
                    "username": "fake_user"
                }
            }
        },
        "command": "train"
    }
    (tmp_path / "run_snapshot.json").write_text(json.dumps(snapshot))

    # Create baseline subdirectory and predictions.csv
    baseline_dir = tmp_path / "baseline"
    baseline_dir.mkdir()
    csv_content = "event_id,ri_prob\ne1,0.12\ne2,0.87\n"
    (baseline_dir / "predictions.csv").write_text(csv_content)

    # Monkeypatch _results_dir to return tmp_path
    monkeypatch.setattr(mcp_server, "_results_dir", lambda: tmp_path)

    return tmp_path


def test_list_results_lists_files_and_sizes(results_dir):
    """Output contains all files with posix paths and formatted byte sizes."""
    output = mcp_server.list_results()

    # Check that all files are listed
    assert "metrics.json" in output
    assert "run_snapshot.json" in output
    assert "baseline/predictions.csv" in output

    # Check that sizes are formatted with commas
    assert "bytes)" in output

    # Verify exact size of metrics.json
    metrics_path = results_dir / "metrics.json"
    metrics_size = metrics_path.stat().st_size
    assert f"metrics.json  ({metrics_size:,} bytes)" in output

    # Check posix-style path for CSV (forward slashes)
    assert "baseline/predictions.csv" in output


def test_read_result_pretty_prints_json(results_dir):
    """read_result pretty-prints JSON with indentation."""
    output = mcp_server.read_result("metrics.json")

    # Should be able to parse it back to the original dict
    parsed = json.loads(output)
    assert parsed == {"roc_auc": 0.61, "pr_auc": 0.44}

    # Should contain indented output (newline + spaces before keys)
    assert '\n  "roc_auc"' in output or '\n  "pr_auc"' in output


def test_read_result_csv_passthrough(results_dir):
    """read_result returns raw CSV content without reformatting."""
    output = mcp_server.read_result("baseline/predictions.csv")

    expected = "event_id,ri_prob\ne1,0.12\ne2,0.87\n"
    assert output == expected


def test_read_result_blocks_path_traversal(results_dir, tmp_path):
    """read_result rejects path traversal attempts."""
    # Create a file outside the results directory to prove it's not read
    outside_file = tmp_path.parent / "outside.json"
    outside_file.write_text('{"secret": "should_not_read"}')

    # Attempt path traversal
    output = mcp_server.read_result("../outside.json")

    assert output == "error: path escapes the results directory"
    # Verify the outside file exists but was not read
    assert outside_file.exists()


def test_read_result_missing_file_returns_clear_error(results_dir):
    """read_result returns clear error for missing files."""
    output = mcp_server.read_result("nope.json")

    assert output.startswith("error: file not found")


def test_read_result_redacts_secrets_and_never_leaks(results_dir):
    """read_result redacts sensitive keys and does not leak secret values."""
    output = mcp_server.read_result("run_snapshot.json")

    # Should contain redaction marker
    assert "***REDACTED***" in output

    # Should NOT contain the actual secret values
    assert "SEGREDO_FAKE_123" not in output
    assert "fake_user" not in output

    # Non-sensitive values should still be present
    assert "train" in output


def test_list_results_missing_dir(tmp_path, monkeypatch):
    """list_results returns appropriate message for nonexistent directory."""
    nonexistent_path = tmp_path / "does_not_exist"
    monkeypatch.setattr(mcp_server, "_results_dir", lambda: nonexistent_path)

    output = mcp_server.list_results()

    assert "results directory does not exist yet:" in output
    assert str(nonexistent_path) in output
