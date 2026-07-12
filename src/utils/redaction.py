# src/utils/redaction.py
"""Credential redaction for configuration mappings.

The canonical implementation lives in the security layer
(security_layer/secret_guard.py) so that all secret handling is maintained
in one place; this module re-exports it for pipeline callers.
"""

from security_layer.secret_guard import (  # noqa: F401
    REDACTION_MARKER,
    SENSITIVE_KEYS,
    redact_secrets,
)
