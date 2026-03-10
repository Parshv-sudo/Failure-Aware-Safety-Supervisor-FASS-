import logging
import math
import os
import sys
import time
import numpy as np
import torch
from typing import List

# Ensure fass_ml is in the path
carla_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
python_api_path = os.path.join(carla_root, "WindowsNoEditor", "PythonAPI")
if python_api_path not in sys.path:
    sys.path.insert(0, python_api_path)

from fass_ml.models.risk_model import FASSRiskNet
from fass_ml.integration.risk_engine import RiskEngine
from fass_ml.training.config import FASSConfig
from fass_ml.data.feature_extractor import FEATURE_DIM, extract_features

from perception.object_detector import Detection
from supervision.risk_model_interface import BaseRiskModel, RiskInput, RiskOutput
from utils.math_utils import clamp

logger = logging.getLogger(__name__)

class FASSIntegratedRiskModel(BaseRiskModel):
    """
    Adapter bridging the `adas_supervision_project` orchestrator and the 
    `fass_ml` neural network safety supervisor.

    This model translates abstract `RiskInput` back into the 30-dim vector
    required by FASSRiskNet, evaluates it via MC-Dropout, and fuses it
    with deterministic safety thresholds.
    """

    def __init__(self, checkpoint_path: str, config: FASSConfig = None):
        self.config = config or FASSConfig()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 1. Load the FASSRiskNet ML Model
        self.model = FASSRiskNet(
            input_dim=FEATURE_DIM, 
            hidden_dims=self.config.hidden_dims, 
            dropout_p=self.config.dropout_p
        ).to(self.device)
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"FASS ML checkpoint not found at {checkpoint_path}")
            
        logger.info(f"Loading FASS ML checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.train()  # MUST BE IN TRAIN MODE FOR MC-DROPOUT!
        
        # 2. Instantiate the FASS Risk Engine (Deterministic Fusion)
        self.risk_engine = RiskEngine(config=self.config)
        self.mc_samples = self.config.mc_samples
        self._ego_state = {}

    def _extract_kinematics(self, ego_vehicle):
        """Extract ego vehicle kinematics, mimicking FASS ML data collection."""
        if not ego_vehicle:
            return {}
            
        vel = ego_vehicle.get_velocity()
        accel = ego_vehicle.get_acceleration()
        ang_vel = ego_vehicle.get_angular_velocity()
        transform = ego_vehicle.get_transform()
        control = ego_vehicle.get_control()
        
        speed = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
        accel_mag = math.sqrt(accel.x**2 + accel.y**2 + accel.z**2)

        pitch = transform.rotation.pitch
        roll = transform.rotation.roll

        now = time.time()
        altitude = transform.location.z
        
        if 'alt_history' not in self._ego_state:
            self._ego_state['alt_history'] = []
        self._ego_state['alt_history'].append((now, altitude))
        self._ego_state['alt_history'] = [(t, a) for t, a in self._ego_state['alt_history'] if now - t < 5.0]
        
        alt_3s_ago = [a for t, a in self._ego_state['alt_history'] if now - t >= 2.5]
        altitude_change = altitude - alt_3s_ago[0] if alt_3s_ago else 0.0

        if speed < 0.5:
            self._ego_state.setdefault('stationary_since', now)
        else:
            self._ego_state['stationary_since'] = now
            
        is_stationary = (now - self._ego_state.get('stationary_since', now)) >= 2.0

        return {
            'speed': round(speed, 4),
            'acceleration': round(accel_mag, 4),
            'yaw_rate': round(ang_vel.z, 4),
            'steer': round(control.steer, 4),
            'throttle': round(control.throttle, 4),
            'brake': round(control.brake, 4),
            'pitch': round(pitch, 2),
            'roll': round(roll, 2),
            'altitude_change': round(altitude_change, 3),
            'is_stationary': is_stationary,
            'is_reversing': bool(control.reverse),
        }

    def compute_risk(self, inputs: RiskInput, raw_detections: List[Detection] = None, ego_speed: float = 0.0, ego_vehicle=None) -> RiskOutput:
        """
        Computes risk using FASS ML and deterministic fusion.
        
        Note: The standard BaseRiskModel signature only takes `inputs`. We 
        allow optional raw data passing for more accurate feature extraction, 
        but fallback to approximating the 30-dim vector if missing.
        """
        # --- 1. Construct the Feature Vector ---
        # We need to approximate the CARLA raw output structures expected by fass_ml's feature_extractor
        
        detected_objects = []
        min_ttc_raw = 999.0
        min_dist_raw = 999.0
        
        if raw_detections:
            for d in raw_detections:
                obj = {
                    'distance': d.distance if hasattr(d, 'distance') else 10.0,
                    'detection_confidence': d.confidence,
                    'speed': 0.0, # Approximate
                    'type': d.class_name if hasattr(d, 'class_name') else 'vehicle'
                }
                detected_objects.append(obj)
                min_dist_raw = min(min_dist_raw, obj['distance'])
                
            # If TTC is provided in normal inputs, approximate a raw TTC
            if inputs.ttc > 0:
                # Reverse normalisation approximation: ttc_norm = (1/TTC) / (1/epsilon) => TTC = epsilon / ttc_norm
                min_ttc_raw = 0.1 / max(inputs.ttc, 1e-6)
        else:
            # If no raw detections are provided, but we have a valid risk input
            # we should ONLY inject a dummy object if the risk input implies danger.
            # E.g. if TTC is finite (<1.0) or confidence is low.
            if inputs.ttc > 0 or inputs.confidence < 1.0:
                min_ttc_raw = 0.1 / max(inputs.ttc, 1e-6) if inputs.ttc > 0 else 999.0
                min_dist_raw = 10.0 # Unknown but presumed close
                detected_objects.append({
                    'distance': min_dist_raw,
                    'detection_confidence': inputs.confidence,
                    'speed': 0.0,
                    'type': 'unknown'
                })
            else:
                # Perfectly safe, no objects
                min_ttc_raw = 999.0
                min_dist_raw = 999.0

        if ego_vehicle:
            ego_kin = self._extract_kinematics(ego_vehicle)
            ego_speed = ego_kin.get('speed', ego_speed)
        else:
            # Fallback approximation from normalized inputs
            ego_kin = {
                'speed': ego_speed,
                'is_stationary': ego_speed < 0.5
            }
        weather = {} # Default good weather
        sensor_health = {'camera_front': True, 'lidar_roof': True, 'radar_front': True} # Default healthy

        feature_vector = extract_features(
            detected_objects=detected_objects,
            ego_kinematics=ego_kin,
            weather=weather,
            sensor_health=sensor_health,
            normalize=True
        )

        if not hasattr(self, '_logged_features'):
            logger.info(f"DEBUG: Actual feature_vector passed to model: {feature_vector}")
            self._logged_features = True

        tensor_x = torch.tensor(feature_vector, dtype=torch.float32).unsqueeze(0).to(self.device)

        # --- 2. MC-Dropout Inference ---
        preds = []
        log_vars = []
        with torch.no_grad():
            for _ in range(self.mc_samples):
                risk_prob, log_var = self.model(tensor_x)
                preds.append(risk_prob.item())
                log_vars.append(log_var.item())
                
        # Calculate Risk and Uncertainty
        ml_risk = float(np.mean(preds))
        
        epistemic_unc = float(np.var(preds))
        aleatoric_unc = float(np.mean(np.exp(log_vars)))
        total_uncertainty = epistemic_unc + aleatoric_unc

        # --- 3. Deterministic Fusion via Risk Engine ---
        self.risk_engine.update(
            ml_risk=ml_risk,
            ml_uncertainty=total_uncertainty,
            min_distance=min_dist_raw,
            min_ttc=min_ttc_raw,
            sensor_failures=0, # Assumed 0 for now
            ego_speed=ego_speed
        )

        fused_risk = self.risk_engine.fused_risk

        # --- 4. Format Output ---
        return RiskOutput(
            risk=fused_risk,
            components={
                "ml_risk": round(ml_risk, 4),
                "epistemic_unc": round(epistemic_unc, 4),
                "aleatoric_unc": round(aleatoric_unc, 4),
                "deterministic_risk": round(self.risk_engine._deterministic_risk, 4),
                "fused_risk": round(fused_risk, 4),
                "override_active": 1.0 if self.risk_engine.is_overriding else 0.0
            }
        )
