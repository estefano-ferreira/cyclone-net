#!/usr/bin/env python3
# security_layer/secret_guard.py
"""
Secret guard: fails when credentials or forbidden files would enter git.

Two layers of defense:
  1. FORBIDDEN FILES — paths that must never be tracked (local credential
     stores and artifacts known to serialize configuration).
  2. SECRET PATTERNS — content scan for credential-shaped strings (API keys,
     passwords, tokens) in tracked/staged text files.

Exit code 0 = clean; 1 = findings (blocks the commit via the pre-commit hook,
fails CI). Standard library only — no dependencies.

Usage:
    python security_layer/secret_guard.py            # scan all tracked files
    python security_layer/secret_guard.py --staged   # scan staged files (hook)
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

# Paths that must never be tracked, no matter their content.
FORBIDDEN_PATHS = [
    re.compile(r"(^|/)config\.yaml$"),          # local config may hold credentials
    re.compile(r"(^|/)run_snapshot\.json$"),    # serializes the runtime config
    re.compile(r"(^|/)\.cdsapirc$"),
    re.compile(r"(^|/)\.netrc$"),
    re.compile(r"(^|/)\.env$"),
    re.compile(r"(^|/)\.copernicusmarine"),
]

# Values that look like placeholders and are allowed.
PLACEHOLDER = re.compile(
    r"^(|\"\"|''|null|none|~|\*\*\*REDACTED\*\*\*|your[_-].*|<.*>|\$\{.*\}|changeme)$",
    re.IGNORECASE,
)

# Credential-shaped content. Each pattern captures the would-be secret value.
# The assignment pattern accepts QUOTED values containing any character except
# the quote itself (real passwords contain '#', '@', etc. — the leaked
# Copernicus password did), and unquoted values that stop at YAML comment/
# structure characters.
ASSIGNMENT_PATTERN = re.compile(
    r"""(?ix)\b(api[_-]?key|apikey|key|password|passwd|secret|token)\b
        \s*[:=]\s*
        (?: "(?P<dq>[^"]{6,})" | '(?P<sq>[^']{6,})' | (?P<bare>[^\s"'#,}{\]\[]{6,}) )"""
)
SECRET_PATTERNS = [
    ASSIGNMENT_PATTERN,
    # UUID (the classic CDS API key shape)
    re.compile(r"(?i)(?P<bare>[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})"),
    # Common provider token prefixes
    re.compile(r"(?P<bare>(sk|ghp|gho|glpat|whsec|xox[bap])[-_][A-Za-z0-9_\-]{16,})"),
]


def find_secret_in_line(line: str) -> str | None:
    """Return the credential-shaped value found in *line*, or None.

    Single decision point shared by the scanner and its tests: applies the
    placeholder allowlist, the code-identifier noise filters, and the
    digit heuristic for assignment-shaped matches.
    """
    for pat in SECRET_PATTERNS:
        m = pat.search(line)
        if not m:
            continue
        groups = m.groupdict()
        val = next((v for v in (groups.get("dq"), groups.get("sq"), groups.get("bare")) if v), None)
        if val is None:
            continue
        val = val.strip()
        if PLACEHOLDER.match(val):
            continue
        # Reduce noise: ignore pure config-key references and paths.
        if val.startswith(("./", "../", "data/", "outputs/", "cfg_get", "cfg[", "self.")):
            continue
        # Assignment-shaped matches must contain a digit to count as a secret:
        # real keys/passwords virtually always do, while code identifiers
        # (e.g. `key = str(path)`) virtually never do. UUID/token-prefix
        # patterns are exempt from this heuristic.
        if pat is ASSIGNMENT_PATTERN and not re.search(r"\d", val):
            continue
        return val
    return None

TEXT_EXTENSIONS = {".py", ".yaml", ".yml", ".json", ".md", ".txt", ".cfg", ".ini",
                   ".toml", ".sh", ".ps1", ".js", ".html", ".css", ".csv", ".gitignore"}

# ---------------------------------------------------------------------------
# Redaction (canonical implementation)
#
# Run snapshots serialize the full runtime configuration for provenance; this
# prevents credentials from ever being written to disk in those snapshots.
# src/utils/redaction.py delegates here so all secret handling lives in the
# security layer.
# ---------------------------------------------------------------------------

REDACTION_MARKER = "***REDACTED***"

SENSITIVE_KEYS = {
    "key", "password", "passwd", "secret", "token",
    "apikey", "api_key", "credentials", "username",
}


def redact_secrets(obj):
    """Return a copy of *obj* with values of sensitive keys replaced.

    Key matching is case-insensitive and exact (no substring matching).
    The input is never mutated.
    """
    if isinstance(obj, dict):
        return {
            k: REDACTION_MARKER if str(k).lower() in SENSITIVE_KEYS else redact_secrets(v)
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [redact_secrets(item) for item in obj]
    return obj

# Files exempt from the CONTENT scan (path rules still apply everywhere):
# the guard's own pattern definitions and its test fixtures contain
# synthetic credential-shaped strings BY DESIGN (they test the scanner).
CONTENT_SCAN_EXEMPT = re.compile(r"^(security_layer/|tests/test_secret_guard\.py$)")


def git_files(staged: bool) -> list[str]:
    cmd = (["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"]
           if staged else ["git", "ls-files"])
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    return [line.strip() for line in out.splitlines() if line.strip()]


def scan(staged: bool) -> int:
    findings: list[str] = []
    files = git_files(staged)

    for rel in files:
        for pat in FORBIDDEN_PATHS:
            if pat.search(rel.replace("\\", "/")):
                findings.append(f"FORBIDDEN FILE {'staged' if staged else 'tracked'}: {rel}")

    for rel in files:
        posix = rel.replace("\\", "/")
        if CONTENT_SCAN_EXEMPT.search(posix):
            continue
        path = Path(rel)
        if not path.exists() or path.suffix.lower() not in TEXT_EXTENSIONS:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            if find_secret_in_line(line) is not None:
                findings.append(f"POSSIBLE SECRET {rel}:{lineno}: {line.strip()[:100]}")

    if findings:
        print("secret_guard: FAILED — potential secrets or forbidden files detected:\n")
        for f in findings:
            print("  " + f)
        print(f"\n{len(findings)} finding(s). Move credentials to ~/.cdsapirc / "
              "copernicusmarine login / environment variables and untrack forbidden files.")
        return 1

    print(f"secret_guard: CLEAN ({len(files)} {'staged' if staged else 'tracked'} files checked)")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staged", action="store_true",
                        help="scan only files staged for commit (pre-commit hook mode)")
    args = parser.parse_args()
    sys.exit(scan(staged=args.staged))


if __name__ == "__main__":
    main()
