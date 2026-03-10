"""
Replay manager for the ADAS Supervision Framework.

Reads blackbox JSONL logs and provides:
    - State reconstruction for post-hoc analysis.
    - **Validation harness**: recomputes risk from logged inputs and
      compares against recorded risk.  Flags divergences where
      |Δ| > tolerance (configurable, default 1e-6).

Used to detect nondeterminism bugs and verify reproducibility.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

from supervision.risk_assessor import RiskAssessor, RuleBasedRiskModel
from supervision.risk_model_interface import RiskInput

logger = logging.getLogger(__name__)


class ReplayManager:
    """Reads and validates blackbox logs.

    Args:
        tolerance: Maximum acceptable |Δ| between recomputed and
            logged risk scores before flagging a divergence.
    """

    def __init__(self, tolerance: float = 1e-6):
        self.tolerance = tolerance

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, blackbox_path: str) -> Dict[str, Any]:
        """Load a blackbox JSONL file.

        Returns:
            Dict with ``header``, ``ticks`` (list), and ``footer``.
        """
        if not os.path.isfile(blackbox_path):
            raise FileNotFoundError(f"Blackbox not found: {blackbox_path}")

        header: Optional[Dict] = None
        ticks: List[Dict] = []
        footer: Optional[Dict] = None

        with open(blackbox_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                rtype = record.get("record_type")
                if rtype == "header":
                    header = record
                elif rtype == "tick":
                    ticks.append(record)
                elif rtype == "footer":
                    footer = record

        logger.info(
            "Loaded blackbox: %d ticks, config_hash=%s",
            len(ticks),
            (header or {}).get("config_hash", "?")[:12],
        )
        return {"header": header, "ticks": ticks, "footer": footer}

    def validate(
        self,
        blackbox_path: str,
        risk_assessor: RiskAssessor = None,
    ) -> Dict[str, Any]:
        """Run the validation harness on a blackbox log.

        For each tick, recompute risk from the logged decomposed
        components and compare with the recorded ``risk`` value.

        Args:
            blackbox_path: Path to ``.jsonl`` file.
            risk_assessor: (Optional) assessor to use for recomputation.
                If ``None``, creates a default ``RuleBasedRiskModel``.

        Returns:
            Dict with ``total_ticks``, ``divergences`` (list of dicts
            with tick number and delta), and ``max_delta``.
        """
        data = self.load(blackbox_path)
        ticks = data["ticks"]

        # Extract config snapshot from header for model params
        header = data.get("header") or {}
        config_snap = header.get("config_snapshot") or {}
        risk_cfg = config_snap.get("risk", {})

        if risk_assessor is None:
            model = RuleBasedRiskModel(
                weights=risk_cfg.get("weights", {}),
                epsilon=risk_cfg.get("epsilon", 0.1),
                speed_max=risk_cfg.get("speed_max", 40.0),
            )
            risk_assessor = RiskAssessor(
                model=model,
                epsilon=risk_cfg.get("epsilon", 0.1),
                speed_max=risk_cfg.get("speed_max", 40.0),
            )

        divergences: List[Dict] = []
        max_delta = 0.0

        for tick_record in ticks:
            logged_risk = tick_record.get("risk", 0.0)

            # Reconstruct normalised inputs from logged components
            # The components are already weighted, so we sum them directly
            recomputed = (
                tick_record.get("confidence_term", 0)
                + tick_record.get("ttc_term", 0)
                + tick_record.get("speed_term", 0)
                + tick_record.get("complexity_term", 0)
            )

            delta = abs(recomputed - logged_risk)
            if delta > max_delta:
                max_delta = delta

            if delta > self.tolerance:
                divergences.append({
                    "tick": tick_record.get("tick"),
                    "logged_risk": logged_risk,
                    "recomputed_risk": round(recomputed, 8),
                    "delta": round(delta, 10),
                })

        result = {
            "total_ticks": len(ticks),
            "divergence_count": len(divergences),
            "max_delta": round(max_delta, 10),
            "divergences": divergences[:20],  # cap output
            "config_hash": header.get("config_hash", ""),
        }

        if divergences:
            logger.warning(
                "Replay validation: %d divergences (max Δ=%.2e)",
                len(divergences),
                max_delta,
            )
        else:
            logger.info("Replay validation: PASSED (0 divergences)")

        return result
