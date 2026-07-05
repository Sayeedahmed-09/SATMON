"""
Run this once in your project folder to generate xgb_model.joblib
and xgb_label_encoder.joblib on your own machine.

Command:
    python train_xgb.py
"""

import pickle
import warnings
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# ── Load your existing RandomForest + feature names ───────────────────────
print("Loading RandomForest model...")
with open("satellite_model.pkl", "rb") as f:
    rf_model = pickle.load(f)
with open("feature_names.pkl", "rb") as f:
    feature_names = pickle.load(f)

print(f"Features: {len(feature_names)}")
print(f"RF classes: {list(rf_model.classes_)}")

# ── Generate synthetic telemetry samples ──────────────────────────────────
# We sample the full plausible range of each feature and label with RF.
# This is knowledge distillation — XGBoost learns the RF's decision surface.
RANGES = {
    "OrbitPhase (%)":          (0,    100),
    "Sunlight (0 or 1)":       (0,    1),
    "BusVoltage (V)":          (10,   35),
    "BusCurrent (A)":          (0,    12),
    "BatteryVoltage (V)":      (2.0,  4.5),
    "BatteryTemperature (°C)": (-15,  90),
    "BatterySOC (%)":          (0,    100),
    "SolarVoltage (V)":        (0,    38),
    "SolarCurrent (A)":        (0,    6),
    "WheelRPM (RPM)":          (-9000,9000),
    "WheelTemperature (°C)":   (-25,  130),
    "CPUUsage (%)":            (0,    100),
    "CPUTemperature (°C)":     (-15,  140),
    "SignalStrength (dBm)":    (-120, -30),
    "GyroMagnitude (deg/s)":   (0,    6),
    "Altitude (km)":           (380,  650),
}

print("Generating 20,000 synthetic samples...")
np.random.seed(42)
N = 20000
X_synth = np.column_stack([
    np.random.uniform(lo, hi, N)
    for name in feature_names
    for lo, hi in [RANGES.get(name, (0, 1))]
])

print("Labelling with RandomForest (knowledge distillation)...")
y_synth = rf_model.predict(X_synth)

classes, counts = np.unique(y_synth, return_counts=True)
print("Class distribution:")
for c, n in zip(classes, counts):
    print(f"  Class {c}: {n} samples")

# ── LabelEncoder (handles non-contiguous class IDs) ───────────────────────
le = LabelEncoder()
y_enc = le.fit_transform(y_synth)
print(f"\nEncoded classes: {le.classes_.tolist()}")

# ── Train XGBoost ─────────────────────────────────────────────────────────
X_tr, X_te, y_tr, y_te = train_test_split(
    X_synth, y_enc, test_size=0.2, random_state=42
)

print("\nTraining XGBoost...")
xgb = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    eval_metric="mlogloss",
    random_state=42,
    verbosity=0,
)
xgb.fit(X_tr, y_tr)

acc = accuracy_score(y_te, xgb.predict(X_te))
print(f"XGBoost accuracy vs RF labels: {acc:.4f}")

# ── Save models ───────────────────────────────────────────────────────────
joblib.dump(xgb, "xgb_model.joblib")
joblib.dump(le,  "xgb_label_encoder.joblib")

print("\nSaved:")
print("  xgb_model.joblib")
print("  xgb_label_encoder.joblib")
print("\nDone. Now run: python app.py")
