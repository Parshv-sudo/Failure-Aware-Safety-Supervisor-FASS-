"""
Control transition module for the ADAS Supervision Framework.

Handles progressive speed reduction and emergency braking during
the takeover sequence.  Generates ``carla.VehicleControl`` commands
that smoothly reduce speed.
"""

import logging

import carla

from utils.math_utils import clamp

logger = logging.getLogger(__name__)


class ControlTransition:
    """Manages gradual control handover from autonomous to manual.

    Args:
        speed_reduction_rate: Target deceleration (m/s²).
    """

    def __init__(self, speed_reduction_rate: float = 2.0):
        self.decel_rate = speed_reduction_rate  # m/s²
        self._last_brake = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def gradual_slow_down(
        self,
        vehicle: "carla.Vehicle",
        delta_t: float,
    ) -> "carla.VehicleControl":
        """Apply gentle braking to progressively reduce speed.

        Args:
            vehicle: Ego ``carla.Vehicle``.
            delta_t: Fixed time-step (seconds).

        Returns:
            The ``carla.VehicleControl`` that was applied.
        """
        speed = self._speed(vehicle)
        if speed < 0.5:
            # Effectively stopped, softly increase hold brake parabolically
            step = 0.005 + (self._last_brake * 0.05)
            brake = min(self._last_brake + step, 0.5)
            ctrl = carla.VehicleControl(throttle=0.0, brake=brake, steer=0.0)
            self._last_brake = brake
        else:
            # Proportional braking
            desired_speed = max(0.0, speed - self.decel_rate * delta_t)
            target_brake = clamp(
                (speed - desired_speed) / max(speed, 1e-6), 0.0, 1.0
            )
            
            # Smoothly ramp the brake parabolically to avoid spikes
            # Base step is small, but grows with current brake pressure (exponential S-curve)
            step = 0.005 + (self._last_brake * 0.05)
            brake_intensity = min(target_brake, self._last_brake + step)
            brake_intensity = clamp(brake_intensity, 0.0, 0.6) # Gentle max brake for stage 2
            
            ctrl = carla.VehicleControl(
                throttle=0.0,
                brake=brake_intensity,
                steer=vehicle.get_control().steer,  # hold lane
            )
            self._last_brake = brake_intensity

        vehicle.apply_control(ctrl)
        return ctrl

    def emergency_brake(self, vehicle: "carla.Vehicle") -> "carla.VehicleControl":
        """Immediate but smooth full braking.

        Args:
            vehicle: Ego ``carla.Vehicle``.

        Returns:
            The ``carla.VehicleControl`` that was applied.
        """
        # Parabolic/exponential curve for emergency stop (fast onset S-curve)
        step = 0.01 + (self._last_brake * 0.1)
        brake = min(self._last_brake + step, 0.8)
        ctrl = carla.VehicleControl(throttle=0.0, brake=brake, steer=0.0)
        vehicle.apply_control(ctrl)
        self._last_brake = brake
        logger.warning("🛑 EMERGENCY BRAKE applied (%.2f).", brake)
        return ctrl

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _speed(vehicle: "carla.Vehicle") -> float:
        """Scalar speed in m/s."""
        v = vehicle.get_velocity()
        return (v.x ** 2 + v.y ** 2 + v.z ** 2) ** 0.5
