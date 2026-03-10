"""
Abstract risk model interface for the ADAS Supervision Framework.

Defines ``BaseRiskModel`` — the contract that every risk model
(rule-based, ML-based, hybrid) must implement.  This enables
future extensibility without changing downstream consumers.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict


@dataclass
class RiskInput:
    """Container for all inputs consumed by a risk model.

    All fields are pre-normalised to [0, 1] by the caller.

    Attributes:
        confidence: Worst (or aggregated) detection confidence.
        ttc: Normalised TTC term (1/max(TTC,ε) scaled by 1/ε).
        speed_factor: Ego speed / speed_max.
        road_complexity: ODD road complexity ∈ [0, 1].
    """
    confidence: float = 1.0
    ttc: float = 0.0
    speed_factor: float = 0.0
    road_complexity: float = 0.0


@dataclass
class RiskOutput:
    """Result produced by a risk model.

    Attributes:
        risk: Scalar risk score ∈ [0, 1].
        components: Breakdown of each weighted term for logging.
    """
    risk: float = 0.0
    components: Dict[str, float] = None

    def __post_init__(self):
        if self.components is None:
            self.components = {}


class BaseRiskModel(ABC):
    """Abstract base class for risk scoring models.

    Sub-classes must implement :meth:`compute_risk` which accepts a
    ``RiskInput`` and returns a ``RiskOutput``.

    The contract guarantees:
        * All ``RiskInput`` fields are in [0, 1].
        * The returned ``risk`` value **must** be in [0, 1].
    """

    @abstractmethod
    def compute_risk(self, inputs: "RiskInput") -> "RiskOutput":
        """Compute a risk score from normalised inputs.

        Args:
            inputs: Pre-normalised risk inputs.

        Returns:
            ``RiskOutput`` with ``risk ∈ [0, 1]`` and a components
            breakdown for forensic logging.
        """
        ...
