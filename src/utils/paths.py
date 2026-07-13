"""Project-root detection and manifest-safe relative paths.

Provenance/result manifests are versioned in a public repository: they must
never record absolute paths from the machine that produced them (username
and directory layout would leak, and the recorded value would be useless on
any other machine). Writers pass paths through ``rel_to_root`` so manifests
carry portable, project-root-relative POSIX strings.
"""
from pathlib import Path

# src/utils/paths.py -> src/utils -> src -> project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def rel_to_root(path: Path) -> str:
    """Best-effort project-root-relative POSIX form of ``path``.

    Falls back to the bare filename when ``path`` lies outside the project
    root (e.g. a custom data dir, or tmp dirs in tests) -- a manifest must
    never record an absolute path.
    """
    p = Path(path).resolve()
    try:
        return p.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return p.name
