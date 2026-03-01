#!/usr/bin/env python
"""
FASS Voting Logic
===================
Combines multiple redundant risk evaluators using weighted voting.

Evaluators:
    1. ML Risk Engine (ml_risk, uncertainty-aware)
    2. TTC-Based Evaluator (physics-based time-to-collision)
    3. Distance-Based Evaluator (proximity threshold)

Voting rules:
    - Each evaluator casts a vote: SAFE / CAUTION / DANGER
    - Weighted voting determines the final advisory
    - DANGER requires majority agreement OR any single evaluator with
      confidence > 0.9 (fail-safe: any strong danger signal triggers)
    - CAUTION requires at least one evaluator voting CAUTION or DANGER

CRITICAL SAFETY RULE:
    A single evaluator can escalate to DANGER unilaterally if its
    confidence exceeds the veto threshold.  This prevents voting
    from masking a genuine threat detected by one sensor modality.

ISO 26262 Note:
    Multi-channel voting with veto authority implements the "voter"
    pattern for ASIL-D safety integrity.
"""

import time
from typing import List, Dict, Optional


# Advisory levels (ordered by severity)
ADVISORY_LEVELS = {'SAFE': 0, 'CAUTION': 1, 'DANGER': 2}
ADVISORY_NAMES = {v: k for k, v in ADVISORY_LEVELS.items()}


class EvaluatorVote:
    """A single evaluator's vote."""

    def __init__(self, name: str, advisory: str, confidence: float, risk_score: float):
        """
        Parameters
        ----------
        name : str
            Evaluator identifier (e.g., 'ML_RISK', 'TTC_PHYSICS', 'DISTANCE').
        advisory : str
            'SAFE', 'CAUTION', or 'DANGER'.
        confidence : float
            Confidence in the vote [0, 1].  For ML, this is 1 - uncertainty.
        risk_score : float
            Raw risk score [0, 1].
        """
        self.name = name
        self.advisory = advisory
        self.confidence = confidence
        self.risk_score = risk_score
        self.level = ADVISORY_LEVELS.get(advisory, 0)

    def to_dict(self):
        return {
            'name': self.name,
            'advisory': self.advisory,
            'confidence': round(self.confidence, 4),
            'risk_score': round(self.risk_score, 4),
        }


class VotingLogic:
    """Weighted multi-evaluator voting with veto authority.

    Safety Rationale:
        Multi-channel voting implements the ISO 26262 "voter" pattern for
        ASIL-D integrity.  Using redundant evaluators (ML, TTC physics,
        distance) ensures no single failure mode can silently suppress a
        danger signal.  The veto mechanism allows any evaluator with high
        confidence to escalate to DANGER unilaterally, preventing a scenario
        where two weak SAFE votes mask one strong DANGER vote.  An empty
        vote set defaults to CAUTION (fail-safe, not fail-operational) to
        guarantee the system never produces SAFE when perception has failed.

    Parameters
    ----------
    weights : dict
        Evaluator name → weight.  Higher weight = more influence.
    veto_threshold : float
        If any evaluator's confidence exceeds this AND votes DANGER,
        the final advisory is DANGER regardless of other votes.
    danger_threshold : float
        Weighted score above this → DANGER.
    caution_threshold : float
        Weighted score above this → CAUTION.
    """

    def __init__(
        self,
        weights: dict = None,
        veto_threshold: float = 0.95,
        danger_threshold: float = 0.75,
        caution_threshold: float = 0.45,
    ):
        self.weights = weights or {
            'ML_RISK': 0.4,
            'TTC_PHYSICS': 0.35,
            'DISTANCE': 0.25,
        }
        self.veto_threshold = veto_threshold
        self.danger_threshold = danger_threshold
        self.caution_threshold = caution_threshold

        self._votes: List[EvaluatorVote] = []
        self._final_advisory = 'SAFE'
        self._veto_triggered = False
        self._last_vote_time = 0.0

    def submit_vote(self, vote: EvaluatorVote):
        """Submit an evaluator's vote."""
        self._votes.append(vote)

    def clear_votes(self):
        """Clear all votes for a new voting round."""
        self._votes.clear()
        self._veto_triggered = False

    def vote(self) -> str:
        """Execute the voting logic and return the final advisory.

        Safety Rationale — Decision Chain:
            1. VETO CHECK (highest priority): Any evaluator voting DANGER
               with confidence ≥ veto_threshold immediately returns DANGER.
               This ensures a single reliable sensor detecting imminent
               collision cannot be outvoted.
            2. WEIGHTED SCORING: Risk scores are aggregated using evaluator
               weights and confidence.  This balances information from all
               channels proportionally.
            3. MAJORITY OVERRIDE: If ≥2 evaluators independently vote
               DANGER, the final result is DANGER regardless of weighted
               score.  This catches cases where moderate individual risks
               compound into systemic danger.

        Returns
        -------
        str : 'SAFE', 'CAUTION', or 'DANGER'.
        """
        self._last_vote_time = time.time()

        if not self._votes:
            # No evaluators → assume CAUTION (fail-safe)
            self._final_advisory = 'CAUTION'
            return self._final_advisory

        # Check for veto (any single evaluator can escalate to DANGER)
        for v in self._votes:
            if v.advisory == 'DANGER' and v.confidence >= self.veto_threshold:
                self._final_advisory = 'DANGER'
                self._veto_triggered = True
                return self._final_advisory

        # Weighted scoring
        total_weight = 0.0
        weighted_risk = 0.0
        for v in self._votes:
            w = self.weights.get(v.name, 0.1)
            weighted_risk += w * v.risk_score * v.confidence
            total_weight += w

        if total_weight > 0:
            normalized_score = weighted_risk / total_weight
        else:
            normalized_score = 0.5  # uncertain → moderate

        # Determine advisory
        if normalized_score >= self.danger_threshold:
            self._final_advisory = 'DANGER'
        elif normalized_score >= self.caution_threshold:
            self._final_advisory = 'CAUTION'
        else:
            self._final_advisory = 'SAFE'

        # Majority check: if 2+ evaluators vote DANGER, override
        danger_count = sum(1 for v in self._votes if v.advisory == 'DANGER')
        if danger_count >= 2:
            self._final_advisory = 'DANGER'

        return self._final_advisory

    @property
    def final_advisory(self) -> str:
        return self._final_advisory

    def get_state(self) -> dict:
        return {
            'final_advisory': self._final_advisory,
            'veto_triggered': self._veto_triggered,
            'votes': [v.to_dict() for v in self._votes],
            'num_evaluators': len(self._votes),
            'timestamp': self._last_vote_time,
        }
