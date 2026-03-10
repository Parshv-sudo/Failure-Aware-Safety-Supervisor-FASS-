"""
Simulation state machine for the ADAS Supervision Framework.

Defines ``SimulationState`` enum and ``SimulationStateMachine`` that
governs the main loop transitions:

    INITIALIZING → RUNNING → TAKEOVER_ACTIVE → FALLBACK_BRAKING → TERMINATED

Invalid transitions are rejected and logged.
"""

import logging
from enum import Enum, auto
from typing import Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class SimulationState(Enum):
    """Possible simulation lifecycle states."""

    INITIALIZING = auto()
    RUNNING = auto()
    TAKEOVER_ACTIVE = auto()
    FALLBACK_BRAKING = auto()
    TERMINATED = auto()


# Valid directed edges in the state graph
_VALID_TRANSITIONS: Dict[SimulationState, Set[SimulationState]] = {
    SimulationState.INITIALIZING: {SimulationState.RUNNING, SimulationState.TERMINATED},
    SimulationState.RUNNING: {
        SimulationState.TAKEOVER_ACTIVE,
        SimulationState.TERMINATED,
    },
    SimulationState.TAKEOVER_ACTIVE: {
        SimulationState.RUNNING,
        SimulationState.FALLBACK_BRAKING,
        SimulationState.TERMINATED,
    },
    SimulationState.FALLBACK_BRAKING: {
        SimulationState.RUNNING,
        SimulationState.TERMINATED,
    },
    SimulationState.TERMINATED: set(),  # terminal
}


class SimulationStateMachine:
    """Manages simulation state with guarded transitions and callbacks.

    Args:
        initial: Starting state (default ``INITIALIZING``).

    Usage::

        sm = SimulationStateMachine()
        sm.on_enter(SimulationState.TAKEOVER_ACTIVE, my_handler)
        sm.transition(SimulationState.RUNNING)
    """

    def __init__(self, initial: SimulationState = SimulationState.INITIALIZING):
        self._state = initial
        self._on_enter: Dict[SimulationState, List[Callable]] = {
            s: [] for s in SimulationState
        }
        self._on_exit: Dict[SimulationState, List[Callable]] = {
            s: [] for s in SimulationState
        }
        self._history: List[SimulationState] = [initial]
        logger.info("State machine initialised → %s", initial.name)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def state(self) -> SimulationState:
        """Current state."""
        return self._state

    @property
    def history(self) -> List[SimulationState]:
        """Ordered list of all states visited."""
        return list(self._history)

    # ------------------------------------------------------------------
    # Transitions
    # ------------------------------------------------------------------

    def transition(self, target: SimulationState) -> bool:
        """Attempt a state transition.

        Args:
            target: Desired next state.

        Returns:
            ``True`` if the transition succeeded.
        """
        if target not in _VALID_TRANSITIONS.get(self._state, set()):
            logger.warning(
                "Invalid transition: %s → %s (ignored)",
                self._state.name,
                target.name,
            )
            return False

        prev = self._state

        # Exit callbacks
        for cb in self._on_exit[prev]:
            cb(prev, target)

        self._state = target
        self._history.append(target)

        # Enter callbacks
        for cb in self._on_enter[target]:
            cb(prev, target)

        logger.info("State transition: %s → %s", prev.name, target.name)
        return True

    def is_terminal(self) -> bool:
        """True if the state machine is in ``TERMINATED``."""
        return self._state == SimulationState.TERMINATED

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def on_enter(self, state: SimulationState, callback: Callable):
        """Register a callback invoked when *state* is entered.

        The callback signature is ``(prev_state, new_state)``.
        """
        self._on_enter[state].append(callback)

    def on_exit(self, state: SimulationState, callback: Callable):
        """Register a callback invoked when *state* is exited."""
        self._on_exit[state].append(callback)
