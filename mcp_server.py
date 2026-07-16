#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CycloneNet — MCP Server

Exposes the CycloneNet pipeline CLI (run.py) as Model Context Protocol tools,
so MCP clients (Claude Code, Claude Desktop, etc.) can drive the pipeline:
prepare, download, preprocess, normalize, train, evaluate, baselines,
causal/spatial validation, data QA, and multi-seed sweeps.

Each tool shells out to `python run.py <command>` in a subprocess. This keeps
the heavy scientific stack (torch, xarray, netCDF4) out of the server process
and guarantees the exact same code path as the CLI, including run snapshots
and leakage-free safeguards.

Usage (stdio transport):
    python mcp_server.py

Client registration example (.mcp.json / claude_desktop_config.json):
    {
      "mcpServers": {
        "cyclonenet": {
          "command": "<venv python>",
          "args": ["<project>/mcp_server.py"]
        }
      }
    }
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Literal

from mcp.server.fastmcp import FastMCP

PROJECT_ROOT = Path(__file__).resolve().parent
RUN_PY = PROJECT_ROOT / "run.py"

# Tail of combined stdout/stderr returned to the client. Pipeline logs can be
# huge (per-event progress lines); the tail carries the summary and errors.
MAX_OUTPUT_CHARS = 20_000

# Generous defaults: downloads and training legitimately run for a long time.
DEFAULT_TIMEOUT_S = 3600
LONG_TIMEOUT_S = 6 * 3600

mcp = FastMCP(
    "cyclonenet",
    instructions=(
        "Tools for the CycloneNet tropical-cyclone rapid-intensification "
        "pipeline. Typical order: prepare -> download_era5 -> preprocess -> "
        "normalize -> train -> evaluate. Ocean-heat validation extras: "
        "download_tchp/preprocess_tchp (TCHP), download_ssh/preprocess_adt "
        "(SLA/ADT), validate_sla, causal_ablation. Long-running tools "
        "(downloads, train, sweep) may take hours; tune timeout_seconds."
    ),
)


async def _run_cli(cli_args: list[str], timeout_seconds: float) -> str:
    """Run `python run.py <cli_args>` and return exit code + output tail."""
    if not RUN_PY.exists():
        return f"error: run.py not found at {RUN_PY}"

    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        str(RUN_PY),
        *cli_args,
        cwd=str(PROJECT_ROOT),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    try:
        raw, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout_seconds)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return (
            f"error: command timed out after {timeout_seconds:.0f}s and was killed: "
            f"run.py {' '.join(cli_args)}. Re-run with a larger timeout_seconds."
        )

    output = raw.decode("utf-8", errors="replace")
    if len(output) > MAX_OUTPUT_CHARS:
        output = (
            f"[output truncated to the last {MAX_OUTPUT_CHARS} characters]\n"
            + output[-MAX_OUTPUT_CHARS:]
        )
    status = "ok" if proc.returncode == 0 else "failed"
    return f"status={status} exit_code={proc.returncode}\ncommand=run.py {' '.join(cli_args)}\n\n{output}"


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

@mcp.tool()
async def prepare(force: bool = False, timeout_seconds: float = DEFAULT_TIMEOUT_S) -> str:
    """Download and prepare the IBTrACS event list (pipeline step 1)."""
    args = ["prepare"] + (["--force"] if force else [])
    return await _run_cli(args, timeout_seconds)


@mcp.tool()
async def download_era5(timeout_seconds: float = LONG_TIMEOUT_S) -> str:
    """Download monthly ERA5 NetCDF files (slow; may take hours)."""
    return await _run_cli(["download-era5"], timeout_seconds)


@mcp.tool()
async def preprocess(timeout_seconds: float = LONG_TIMEOUT_S) -> str:
    """Extract per-event data cubes and metadata from ERA5 files."""
    return await _run_cli(["preprocess"], timeout_seconds)


@mcp.tool()
async def normalize(timeout_seconds: float = DEFAULT_TIMEOUT_S) -> str:
    """Audit events, build the valid manifest, create leakage-free storm-based
    splits, and compute train-only normalization statistics."""
    return await _run_cli(["normalize"], timeout_seconds)


@mcp.tool()
async def train(timeout_seconds: float = LONG_TIMEOUT_S) -> str:
    """Train the CycloneNet model (requires `normalize` first;
    may take hours)."""
    return await _run_cli(["train"], timeout_seconds)


@mcp.tool()
async def evaluate(
    split: Literal["val", "test"] = "test",
    calibrate: bool = False,
    spatial: bool = False,
    timeout_seconds: float = DEFAULT_TIMEOUT_S,
) -> str:
    """Evaluate the trained model on a split. Set spatial=True to also run the
    TCHP spatial validation (requires `preprocess_tchp` to have enriched the
    event metadata)."""
    args = ["evaluate", "--split", split]
    if calibrate:
        args.append("--calibrate")
    if spatial:
        args.append("--spatial")
    return await _run_cli(args, timeout_seconds)


@mcp.tool()
async def baseline(
    split: Literal["val", "test"] = "test",
    timeout_seconds: float = DEFAULT_TIMEOUT_S,
) -> str:
    """Train and evaluate the tabular logistic-regression baseline."""
    return await _run_cli(["baseline", "--split", split], timeout_seconds)


@mcp.tool()
async def sweep(seeds: str = "0,1,2", timeout_seconds: float = LONG_TIMEOUT_S) -> str:
    """Run a multi-seed training sweep. `seeds` is a comma-separated list,
    e.g. "0,1,2" (may take many hours)."""
    return await _run_cli(["sweep", "--seeds", seeds], timeout_seconds)


# ---------------------------------------------------------------------------
# Ocean-heat data (TCHP / SLA / ADT)
# ---------------------------------------------------------------------------

@mcp.tool()
async def download_tchp(force: bool = False, timeout_seconds: float = LONG_TIMEOUT_S) -> str:
    """Download TCHP (Tropical Cyclone Heat Potential) validation data from
    NOAA/AOML ERDDAP with Copernicus fallback. Note: gridded TCHP only exists
    publicly for 2022 onward."""
    args = ["download-tchp"] + (["--force"] if force else [])
    return await _run_cli(args, timeout_seconds)


@mcp.tool()
async def preprocess_tchp(timeout_seconds: float = DEFAULT_TIMEOUT_S) -> str:
    """Enrich event metadata with audited TCHP peak locations (required by
    evaluate(spatial=True))."""
    return await _run_cli(["preprocess-tchp"], timeout_seconds)


@mcp.tool()
async def download_ssh(force: bool = False, timeout_seconds: float = LONG_TIMEOUT_S) -> str:
    """Download Copernicus sea-level data (SLA and ADT), the surface signature
    of subsurface ocean heat content."""
    args = ["download-ssh"] + (["--force"] if force else [])
    return await _run_cli(args, timeout_seconds)


@mcp.tool()
async def preprocess_adt(timeout_seconds: float = DEFAULT_TIMEOUT_S) -> str:
    """Sample the ADT ocean channel onto each event grid and compute
    train-only ADT normalization statistics (enables model.use_adt_input)."""
    return await _run_cli(["preprocess-adt"], timeout_seconds)


# ---------------------------------------------------------------------------
# Scientific validation
# ---------------------------------------------------------------------------

@mcp.tool()
async def validate_sla(
    year: int = 2023,
    window_deg: float = 5.0,
    max_events: int | None = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_S,
) -> str:
    """Observational validation of SLA/ADT against TCHP and RI: proxy
    correlation, peak co-location, and RI-vs-non-RI comparison for one year."""
    args = ["validate-sla", "--year", str(year), "--window-deg", str(window_deg)]
    if max_events is not None:
        args += ["--max-events", str(max_events)]
    return await _run_cli(args, timeout_seconds)


@mcp.tool()
async def causal_ablation(
    split: Literal["val", "test"] = "test",
    k: float = 0.05,
    factor: float = 0.5,
    channels: list[str] | None = None,
    max_events: int | None = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_S,
) -> str:
    """Counterfactual FuelMap ablation test: ablate the model's predicted fuel
    region vs a low-fuel control region and compare the drop in predicted RI
    probability (paired t-test). k = top/bottom-k mask fraction, factor =
    ablation strength in [0, 1]."""
    args = ["causal", "--split", split, "--k", str(k), "--factor", str(factor)]
    args += ["--channels"] + (channels or ["sst_anom_K", "wind_mps"])
    if max_events is not None:
        args += ["--max-events", str(max_events)]
    return await _run_cli(args, timeout_seconds)


@mcp.tool()
async def dataqa(
    split: Literal["train", "val", "test"] = "test",
    timeout_seconds: float = DEFAULT_TIMEOUT_S,
) -> str:
    """Run artifact quality assurance on a split and save a JSON report."""
    return await _run_cli(["dataqa", "--split", split], timeout_seconds)


# ---------------------------------------------------------------------------
# Results inspection (read-only)
# ---------------------------------------------------------------------------

SENSITIVE_KEYS = {
    "key", "password", "passwd", "secret", "token",
    "apikey", "api_key", "credentials", "username",
}


def _redact_secrets(obj):
    """Recursively replace values of sensitive keys (case-insensitive) with
    a redaction marker. Returns a new structure; does not mutate the input."""
    if isinstance(obj, dict):
        return {
            k: "***REDACTED***" if str(k).lower() in SENSITIVE_KEYS else _redact_secrets(v)
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [_redact_secrets(item) for item in obj]
    return obj


def _results_dir() -> Path:
    """Resolve paths.results_dir from config.yaml without importing src/."""
    import yaml

    cfg_path = PROJECT_ROOT / "config.yaml"
    results = "./results"
    if cfg_path.exists():
        with cfg_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        results = (cfg.get("paths") or {}).get("results_dir", results)
    return (PROJECT_ROOT / results).resolve()


@mcp.tool()
def list_results() -> str:
    """List result artifacts (JSON/CSV reports, metrics, checkpoints) under the
    configured results directory, relative paths with sizes."""
    root = _results_dir()
    if not root.exists():
        return f"results directory does not exist yet: {root}"
    lines = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            rel = path.relative_to(root)
            lines.append(f"{rel.as_posix()}  ({path.stat().st_size:,} bytes)")
    return f"results_dir={root}\n" + ("\n".join(lines) if lines else "(empty)")


@mcp.tool()
def read_result(relative_path: str) -> str:
    """Read a result file (JSON/CSV/TXT) by path relative to the results
    directory, e.g. "causal/causal_ablation_test.json". Values of sensitive keys
    (credentials, tokens, API keys) are redacted in JSON output."""
    root = _results_dir()
    target = (root / relative_path).resolve()
    if not target.is_relative_to(root):
        return "error: path escapes the results directory"
    if not target.exists():
        return f"error: file not found: {relative_path}"
    if target.stat().st_size > 2_000_000:
        return f"error: file too large to return ({target.stat().st_size:,} bytes)"
    text = target.read_text(encoding="utf-8", errors="replace")
    if target.suffix == ".json":
        try:
            return json.dumps(_redact_secrets(json.loads(text)), indent=2)
        except json.JSONDecodeError:
            pass
    return text


if __name__ == "__main__":
    mcp.run()
