import sys
import os

sys.path.append(os.path.abspath(r'C:\Users\parsh\Desktop\CARLA 0.9.15\WindowsNoEditor\PythonAPI'))
sys.path.append(os.path.abspath(r'C:\Users\parsh\Desktop\CARLA 0.9.15\adas_supervision_project'))

from fass_ml.data.feature_extractor import extract_features

print("Testing Extract Features with empty arrays...")

feat = extract_features(
    detected_objects=[],
    ego_kinematics={'speed': 0.0, 'is_stationary': True},
    weather={},
    sensor_health={},
    normalize=True
)

for i, x in enumerate(feat):
    print(f"Index {i}: {x:.4f}")

from supervision.fass_integrated_risk_model import FASSIntegratedRiskModel
from supervision.risk_assessor import RiskInput

model = FASSIntegratedRiskModel(checkpoint_path=r'C:\Users\parsh\Desktop\CARLA 0.9.15\WindowsNoEditor\PythonAPI\fass_checkpoints\best_model.pt')
inputs = RiskInput(confidence=1.0, ttc=0.0, speed_factor=0.0, road_complexity=0.0)
out = model.compute_risk(inputs, raw_detections=[], ego_speed=0.0)
print("COMPONENTS:", out.components)

