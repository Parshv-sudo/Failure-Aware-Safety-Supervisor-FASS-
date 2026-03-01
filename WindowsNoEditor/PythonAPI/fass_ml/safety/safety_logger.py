#!/usr/bin/env python
"""
FASS Safety Logger
====================
Structured JSON logging for all safety-critical events, predictions,
and interventions.

Log Categories:
    PREDICTION     — every ML prediction (risk, uncertainty, advisory)
    CRITICAL_EVENT — danger advisories, interventions, overrides
    OVERRIDE       — deterministic threshold overrides of ML
    INTERVENTION   — vehicle control interventions (brake, speed limit)
    SYSTEM         — module initialization, errors, latency warnings

Each log entry includes:
    - ISO 8601 timestamp
    - Monotonic tick counter
    - Module source (e.g., 'RiskEngine', 'VotingLogic')
    - Severity level (INFO, WARNING, CRITICAL)
    - Scenario ID / tags for traceability
    - Epistemic and aleatoric uncertainty (decomposed)
    - Full context payload (JSON)

ISO 26262 Note:
    Safety logs are the primary evidence for post-incident analysis.
    They must be tamper-evident (append-only) and include sufficient
    context to reconstruct the full decision chain.
"""

import os
import json
import time
import logging
from datetime import datetime, timezone
from typing import Optional


class SafetyLogger:
    """Structured safety event logger.

    Writes JSON-lines (.jsonl) files for machine-parseable analysis.
    Also uses Python logging for human-readable console output.

    Parameters
    ----------
    log_dir : str
        Directory for log files.
    console_level : str
        Minimum level for console output ('INFO', 'WARNING', 'CRITICAL').
    """

    def __init__(
        self,
        log_dir: str = './fass_logs',
        console_level: str = 'WARNING',
    ):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        # Create log files
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self._prediction_file = os.path.join(log_dir, f'predictions_{timestamp}.jsonl')
        self._critical_file = os.path.join(log_dir, f'critical_events_{timestamp}.jsonl')
        self._intervention_file = os.path.join(log_dir, f'interventions_{timestamp}.jsonl')

        # Python logger for console
        self._logger = logging.getLogger('FASS_Safety')
        if not self._logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(getattr(logging, console_level, logging.WARNING))
            fmt = logging.Formatter(
                '[%(asctime)s] [FASS:%(levelname)s] %(message)s',
                datefmt='%H:%M:%S'
            )
            handler.setFormatter(fmt)
            self._logger.addHandler(handler)
            self._logger.setLevel(logging.DEBUG)

        self._event_counter = 0

    def _make_entry(self, category: str, severity: str, data: dict,
                    scenario_id: str = None) -> dict:
        """Create a structured log entry."""
        self._event_counter += 1
        entry = {
            'event_id': self._event_counter,
            'timestamp_utc': datetime.now(timezone.utc).isoformat(),
            'timestamp_mono': time.monotonic(),
            'category': category,
            'severity': severity,
            'scenario_id': scenario_id or data.get('scenario_id', 'unknown'),
        }

        # Extract uncertainty decomposition if available
        if 'epistemic_unc' in data:
            entry['epistemic_uncertainty'] = data['epistemic_unc']
        if 'aleatoric_unc' in data:
            entry['aleatoric_uncertainty'] = data['aleatoric_unc']

        entry['data'] = data
        return entry

    def _write_jsonl(self, filepath: str, entry: dict):
        """Append a JSON entry to a .jsonl file."""
        with open(filepath, 'a') as f:
            f.write(json.dumps(entry, default=str) + '\n')

    def log_prediction(self, data: dict, scenario_id: str = None):
        """Log a routine ML prediction."""
        entry = self._make_entry('PREDICTION', 'INFO', data, scenario_id)
        self._write_jsonl(self._prediction_file, entry)

    def log_critical_event(self, data: dict, scenario_id: str = None):
        """Log a safety-critical event (DANGER advisory or intervention)."""
        entry = self._make_entry('CRITICAL_EVENT', 'CRITICAL', data, scenario_id)
        self._write_jsonl(self._critical_file, entry)
        self._logger.critical(
            f"CRITICAL EVENT: advisory={data.get('final_advisory', '?')} "
            f"risk={data.get('fused_risk', '?')} "
            f"intervention={data.get('intervention', {}).get('type', '?')} "
            f"scenario={scenario_id or data.get('scenario_id', '?')}"
        )

    def log_override(self, data: dict, scenario_id: str = None):
        """Log a deterministic override of ML prediction."""
        entry = self._make_entry('OVERRIDE', 'WARNING', data, scenario_id)
        self._write_jsonl(self._critical_file, entry)
        self._logger.warning(
            f"DETERMINISTIC OVERRIDE: reason={data.get('reason', '?')} "
            f"ml_risk={data.get('ml_risk', '?')} "
            f"scenario={scenario_id or data.get('scenario_id', '?')}"
        )

    def log_intervention(self, data: dict, scenario_id: str = None):
        """Log a vehicle intervention."""
        entry = self._make_entry('INTERVENTION', 'CRITICAL', data, scenario_id)
        self._write_jsonl(self._intervention_file, entry)
        self._logger.critical(
            f"INTERVENTION: type={data.get('type', '?')} "
            f"brake={data.get('brake_force', 0)} "
            f"scenario={scenario_id or data.get('scenario_id', '?')}"
        )

    def log_system(self, message: str, severity: str = 'INFO'):
        """Log a system-level message."""
        entry = self._make_entry('SYSTEM', severity, {'message': message})
        self._write_jsonl(self._critical_file, entry)
        getattr(self._logger, severity.lower(), self._logger.info)(
            f"SYSTEM: {message}"
        )

    @property
    def log_files(self) -> dict:
        """Return paths to all log files."""
        return {
            'predictions': self._prediction_file,
            'critical_events': self._critical_file,
            'interventions': self._intervention_file,
        }

    @property
    def event_count(self) -> int:
        return self._event_counter
