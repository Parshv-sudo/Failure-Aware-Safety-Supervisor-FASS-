"""
Event logger for the ADAS Supervision Framework.

Logs discrete events (collisions, lane invasions, takeover
transitions, state-machine changes) as structured JSON records
to an append-only event log file.
"""

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict

logger = logging.getLogger(__name__)


class EventLogger:
    """Structured event logger.

    Each event is a single-line JSON object appended to the log file.

    Args:
        output_dir: Directory to write log files into.
        simulation_id: Unique run identifier (UUID string).
    """

    def __init__(self, output_dir: str, simulation_id: str):
        os.makedirs(output_dir, exist_ok=True)
        self._path = os.path.join(output_dir, f"{simulation_id}_events.jsonl")
        self._fh = open(self._path, "a", encoding="utf-8")
        logger.info("Event log: %s", self._path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log(self, event_type: str, data: Dict[str, Any] = None):
        """Write an event record.

        Args:
            event_type: Short label (e.g. ``"collision"``,
                ``"lane_invasion"``, ``"takeover_stage_change"``).
            data: Arbitrary key-value payload.
        """
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "data": data or {},
        }
        self._fh.write(json.dumps(record) + "\n")
        self._fh.flush()

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self):
        """Flush and close the log file."""
        if self._fh and not self._fh.closed:
            self._fh.flush()
            self._fh.close()
            logger.info("Event log closed.")
