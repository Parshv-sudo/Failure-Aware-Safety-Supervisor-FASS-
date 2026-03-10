"""
Forensic blackbox recorder for the ADAS Supervision Framework.

Writes a JSON-lines (.jsonl) file with:
  1. A **header record** containing simulation metadata (UUID,
     config hash, CARLA version, config snapshot).
  2. Continuous **per-tick records** with decomposed risk components,
     vehicle state, detections, and supervision status.

The recorder respects a configurable ``log_every_n_ticks`` to
control file size at high simulation frequencies.
"""

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict

logger = logging.getLogger(__name__)


class BlackboxRecorder:
    """Forensic-grade flight-data recorder.

    Args:
        output_dir: Directory to write the JSONL file.
        simulation_id: Unique run identifier.
        config_hash: SHA-256 of the config file.
        config_snapshot: Full parsed config dict.
        scenario_name: Active scenario label.
        log_every_n_ticks: Write a tick record every *N* ticks.
    """

    def __init__(
        self,
        output_dir: str,
        simulation_id: str,
        config_hash: str,
        config_snapshot: dict,
        scenario_name: str = "",
        log_every_n_ticks: int = 1,
    ):
        os.makedirs(output_dir, exist_ok=True)
        self._path = os.path.join(output_dir, f"{simulation_id}_blackbox.jsonl")
        self._fh = open(self._path, "w", encoding="utf-8")
        self._tick_counter = 0
        self._log_every = max(log_every_n_ticks, 1)

        # Write forensic header
        header = {
            "record_type": "header",
            "simulation_id": simulation_id,
            "timestamp_start": datetime.now(timezone.utc).isoformat(),
            "config_hash": config_hash,
            "carla_version": "0.9.15",
            "scenario": scenario_name,
            "config_snapshot": config_snapshot,
        }
        self._fh.write(json.dumps(header) + "\n")
        self._fh.flush()
        logger.info("Blackbox recorder: %s", self._path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_tick(self, data: Dict[str, Any]):
        """Record a single simulation tick.

        The record is only actually written if the internal tick
        counter is a multiple of ``log_every_n_ticks``.

        Args:
            data: Dictionary containing at minimum:
                ``timestamp``, ``position``, ``speed``, ``steering``,
                ``risk``, ``confidence_term``, ``ttc_term``,
                ``speed_term``, ``complexity_term``.
        """
        self._tick_counter += 1
        if self._tick_counter % self._log_every != 0:
            return

        record = {"record_type": "tick", **data}
        self._fh.write(json.dumps(record) + "\n")

    def flush(self):
        """Force-flush buffered data to disk."""
        if self._fh and not self._fh.closed:
            self._fh.flush()

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self):
        """Write a footer and close the file."""
        footer = {
            "record_type": "footer",
            "timestamp_end": datetime.now(timezone.utc).isoformat(),
            "total_ticks_recorded": self._tick_counter,
        }
        self._fh.write(json.dumps(footer) + "\n")
        self._fh.flush()
        self._fh.close()
        logger.info(
            "Blackbox closed — %d ticks recorded.", self._tick_counter
        )
