"""Transparent expert-system gating for forensic RI diagnostics.

This module implements physics-motivated plausibility constraints that can
be applied AFTER the neural model outputs a probability.

Scientific requirements:
- Rules MUST be configured via config.yaml and frozen prior to test evaluation.
- Report ablations: model-only, rules-only, and hybrid.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class ExpertDecision:
    allow_positive: bool
    weight_multiplier: float
    reason: str


class ExpertSystem:
    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg or {}

    def apply(self, proba: float, wind_knots: Optional[float], pressure_mb: Optional[float]) -> ExpertDecision:
        if not bool(self.cfg.get("enabled", False)):
            return ExpertDecision(True, 1.0, "disabled")

        if wind_knots is None or pressure_mb is None:
            return ExpertDecision(True, 1.0, "missing_meta")

        basic = self.cfg.get("basic_gate", {}) or {}
        max_p = float(basic.get("max_pressure_mb", 1000.0))
        min_w = float(basic.get("min_wind_knots", 35.0))
        mult = float(basic.get("weight_multiplier", 0.1))

        if pressure_mb > max_p or wind_knots < min_w:
            return ExpertDecision(False, mult, f"basic_gate(p>{max_p} or w<{min_w})")

        secondary = self.cfg.get("secondary_gate", {}) or {}
        if bool(secondary.get("enabled", False)):
            min_proba = float(secondary.get("min_proba", 0.98))
            max_p2 = float(secondary.get("max_pressure_mb", 980.0))
            min_w2 = float(secondary.get("min_wind_knots", 65.0))
            mult2 = float(secondary.get("weight_multiplier", 0.3))

            if proba < min_proba and pressure_mb > max_p2 and wind_knots < min_w2:
                return ExpertDecision(False, mult2, "secondary_gate(conservative_filter)")

        return ExpertDecision(True, 1.0, "pass")
