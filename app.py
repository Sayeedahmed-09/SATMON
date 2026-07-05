import os, json, pickle, random, warnings
warnings.filterwarnings('ignore')
from datetime import datetime, timezone

import numpy as np
import joblib
import shap
from lime.lime_tabular import LimeTabularExplainer
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Load models ───────────────────────────────────────────────────────────
xgb_model = joblib.load(os.path.join(BASE_DIR, "xgb_model.joblib"))
xgb_le    = joblib.load(os.path.join(BASE_DIR, "xgb_label_encoder.joblib"))
shap_explainer = shap.TreeExplainer(xgb_model)

with open(os.path.join(BASE_DIR, "satellite_model.pkl"), "rb") as f:
    rf_model = pickle.load(f)
with open(os.path.join(BASE_DIR, "feature_names.pkl"), "rb") as f:
    feature_names = pickle.load(f)

# ── LIME explainer (fitted on synthetic background data) ──────────────────
_bg_seed = np.random.RandomState(42)
_bg_data  = np.column_stack([
    _bg_seed.uniform(lo, hi, 500)
    for lo, hi in [
        (0,100),(0,1),(10,35),(0,12),(2,4.5),(-15,90),(0,100),
        (0,38),(0,6),(-9000,9000),(-25,130),(0,100),(-15,140),
        (-120,-30),(0,6),(380,650)
    ]
])
lime_explainer = LimeTabularExplainer(
    _bg_data,
    feature_names=feature_names,
    class_names=[
        "Normal Operation", "Power Anomaly", "Thermal Anomaly",
        "Critical Fault",   "Communication Issue",
        "Attitude Control Anomaly", "CPU / System Overload",
    ],
    mode='classification',
    discretize_continuous=True,
    random_state=42,
)

# ── Class definitions ─────────────────────────────────────────────────────
CLASS_LABELS = {
    0:"Normal Operation", 1:"Power Anomaly", 2:"Thermal Anomaly",
    3:"Critical Fault",   4:"Communication Issue",
    5:"Attitude Control Anomaly", 6:"CPU / System Overload"
}
CLASS_SEVERITY = {
    0:"nominal",1:"warning",2:"warning",3:"critical",
    4:"warning",5:"warning",6:"warning"
}
CLASS_RECOMMENDATIONS = {
    0:["All subsystems nominal — no action required.",
       "Continue routine telemetry monitoring on standard cadence.",
       "Next scheduled health check is on track."],
    1:["Switch non-critical loads to low-power mode to reduce bus current draw.",
       "Verify solar panel orientation and confirm sun-pointing attitude.",
       "Check battery charge controller for faults if SOC continues to drop.",
       "Reduce payload duty cycle until bus voltage stabilises."],
    2:["Activate thermal control subsystem (heaters/radiators) as appropriate.",
       "Reduce CPU and payload duty cycle to lower internal heat generation.",
       "Reorient satellite to reduce direct solar thermal load if persistent.",
       "Monitor battery temperature closely — thermal runaway risk above 60°C."],
    3:["SAFE MODE recommended — multiple subsystems degraded.",
       "Prioritise power and thermal stabilisation before resuming normal ops.",
       "Schedule ground station contact for manual diagnostic override.",
       "Disable non-essential subsystems to conserve power margin."],
    4:["Re-point high-gain antenna toward ground station.",
       "Switch to backup communication frequency/transponder if available.",
       "Increase transmission power within allowed budget.",
       "Log link outage duration for post-pass analysis."],
    5:["Run reaction wheel desaturation routine via magnetorquers.",
       "Cross-check gyroscope readings against star tracker for sensor drift.",
       "Switch to backup attitude control sensor set if anomaly persists.",
       "Reduce wheel RPM setpoint to lower mechanical stress."],
    6:["Restart non-essential background processes to free CPU headroom.",
       "Throttle payload data processing rate temporarily.",
       "Monitor CPU temperature — overload can compound thermal issues.",
       "If CPU temperature exceeds 90°C initiate emergency CPU throttle."],
}
CLASS_EXPLAIN = {
    0:"All 16 telemetry parameters fell within expected nominal envelopes. SHAP values were near-zero across all features confirming no single parameter dominated the Normal classification.",
    1:"Power-chain indicators — bus voltage, bus current, battery SOC, and solar input — collectively drove this classification. SHAP analysis shows these features pushed the prediction strongly toward Power Anomaly.",
    2:"Elevated temperature readings across battery, wheel, and CPU subsystems exceeded nominal ceilings. Thermal features showed the largest positive SHAP contributions to this prediction.",
    3:"Multiple subsystem groups simultaneously showed extreme out-of-range values. SHAP reveals overlapping fault signatures across power, thermal, and attitude subsystems.",
    4:"All telemetry appeared nominal except signal strength which fell far below the usable floor. The communications feature dominated this prediction with a large positive SHAP value.",
    5:"Gyroscope magnitude and wheel RPM deviated significantly from nominal attitude-control ranges. Attitude subsystem features were the primary positive SHAP contributors.",
    6:"CPU usage and CPU temperature both exceeded safe operating thresholds. On-board computing features were the dominant contributors with the highest positive SHAP values.",
}
NOMINAL_RANGES = {
    "OrbitPhase (%)":          {"min":0,    "max":100,  "unit":"%",    "critical_min":0,   "critical_max":100},
    "Sunlight (0 or 1)":       {"min":0,    "max":1,    "unit":"",     "critical_min":0,   "critical_max":1},
    "BusVoltage (V)":          {"min":26,   "max":31,   "unit":"V",    "critical_min":20,  "critical_max":34},
    "BusCurrent (A)":          {"min":0,    "max":6,    "unit":"A",    "critical_min":0,   "critical_max":10},
    "BatteryVoltage (V)":      {"min":3.2,  "max":4.2,  "unit":"V",    "critical_min":2.8, "critical_max":4.4},
    "BatteryTemperature (°C)": {"min":-10,  "max":45,   "unit":"°C",   "critical_min":-20, "critical_max":60},
    "BatterySOC (%)":          {"min":20,   "max":100,  "unit":"%",    "critical_min":10,  "critical_max":100},
    "SolarVoltage (V)":        {"min":0,    "max":36,   "unit":"V",    "critical_min":0,   "critical_max":38},
    "SolarCurrent (A)":        {"min":0,    "max":5,    "unit":"A",    "critical_min":0,   "critical_max":6},
    "WheelRPM (RPM)":          {"min":-8000,"max":8000, "unit":"RPM",  "critical_min":-9000,"critical_max":9000},
    "WheelTemperature (°C)":   {"min":-20,  "max":70,   "unit":"°C",   "critical_min":-25, "critical_max":90},
    "CPUUsage (%)":            {"min":0,    "max":100,  "unit":"%",    "critical_min":0,   "critical_max":100},
    "CPUTemperature (°C)":     {"min":-10,  "max":85,   "unit":"°C",   "critical_min":-15, "critical_max":100},
    "SignalStrength (dBm)":    {"min":-110, "max":-40,  "unit":"dBm",  "critical_min":-115,"critical_max":-35},
    "GyroMagnitude (deg/s)":   {"min":0,    "max":5,    "unit":"deg/s","critical_min":0,   "critical_max":7},
    "Altitude (km)":           {"min":400,  "max":600,  "unit":"km",   "critical_min":380, "critical_max":650},
}
SUBSYSTEMS = {
    "power":    {"label":"Power & Electrical","icon":"⚡","features":["BusVoltage (V)","BusCurrent (A)","BatteryVoltage (V)","BatteryTemperature (°C)","BatterySOC (%)","SolarVoltage (V)","SolarCurrent (A)"]},
    "thermal":  {"label":"Thermal",           "icon":"🌡️","features":["BatteryTemperature (°C)","WheelTemperature (°C)","CPUTemperature (°C)"]},
    "attitude": {"label":"Attitude Control",  "icon":"🌀","features":["WheelRPM (RPM)","GyroMagnitude (deg/s)"]},
    "compute":  {"label":"On-Board Computing","icon":"🖥️","features":["CPUUsage (%)","CPUTemperature (°C)"]},
    "comms":    {"label":"Communications",    "icon":"📡","features":["SignalStrength (dBm)"]},
    "orbital":  {"label":"Orbital",           "icon":"🛰️","features":["OrbitPhase (%)","Altitude (km)","Sunlight (0 or 1)"]},
}
PRESET_SCENARIOS = {
    "nominal":       {"label":"Normal Operation","tier":"high","severity":"nominal","desc":"All 16 parameters within nominal range — expect confident Normal classification.","values":{"OrbitPhase (%)":45,"Sunlight (0 or 1)":1,"BusVoltage (V)":28.5,"BusCurrent (A)":3.2,"BatteryVoltage (V)":3.8,"BatteryTemperature (°C)":22,"BatterySOC (%)":88,"SolarVoltage (V)":32,"SolarCurrent (A)":2.6,"WheelRPM (RPM)":4800,"WheelTemperature (°C)":32,"CPUUsage (%)":38,"CPUTemperature (°C)":52,"SignalStrength (dBm)":-75,"GyroMagnitude (deg/s)":0.04,"Altitude (km)":500}},
    "power_fault":   {"label":"Power Anomaly","tier":"high","severity":"warning","desc":"Low bus voltage, depleted battery, no solar input.","values":{"OrbitPhase (%)":60,"Sunlight (0 or 1)":0,"BusVoltage (V)":21.5,"BusCurrent (A)":7.8,"BatteryVoltage (V)":3.1,"BatteryTemperature (°C)":30,"BatterySOC (%)":18,"SolarVoltage (V)":4,"SolarCurrent (A)":0.3,"WheelRPM (RPM)":4500,"WheelTemperature (°C)":34,"CPUUsage (%)":45,"CPUTemperature (°C)":58,"SignalStrength (dBm)":-82,"GyroMagnitude (deg/s)":0.06,"Altitude (km)":498}},
    "thermal_fault": {"label":"Thermal Anomaly","tier":"high","severity":"warning","desc":"Battery, wheel and CPU temperatures all above nominal ceilings.","values":{"OrbitPhase (%)":30,"Sunlight (0 or 1)":1,"BusVoltage (V)":28,"BusCurrent (A)":3.5,"BatteryVoltage (V)":3.6,"BatteryTemperature (°C)":68,"BatterySOC (%)":70,"SolarVoltage (V)":33,"SolarCurrent (A)":2.8,"WheelRPM (RPM)":5200,"WheelTemperature (°C)":92,"CPUUsage (%)":55,"CPUTemperature (°C)":110,"SignalStrength (dBm)":-78,"GyroMagnitude (deg/s)":0.07,"Altitude (km)":501}},
    "critical_fault":{"label":"Critical Fault","tier":"high","severity":"critical","desc":"Multiple systems failing simultaneously.","values":{"OrbitPhase (%)":15,"Sunlight (0 or 1)":0,"BusVoltage (V)":14,"BusCurrent (A)":9.5,"BatteryVoltage (V)":2.6,"BatteryTemperature (°C)":75,"BatterySOC (%)":5,"SolarVoltage (V)":0.5,"SolarCurrent (A)":0,"WheelRPM (RPM)":-500,"WheelTemperature (°C)":95,"CPUUsage (%)":97,"CPUTemperature (°C)":118,"SignalStrength (dBm)":-108,"GyroMagnitude (deg/s)":3.2,"Altitude (km)":470}},
    "comm_fault":    {"label":"Communication Issue","tier":"high","severity":"warning","desc":"Everything nominal except signal strength far below usable floor.","values":{"OrbitPhase (%)":80,"Sunlight (0 or 1)":1,"BusVoltage (V)":28.2,"BusCurrent (A)":3,"BatteryVoltage (V)":3.9,"BatteryTemperature (°C)":20,"BatterySOC (%)":92,"SolarVoltage (V)":31.5,"SolarCurrent (A)":2.4,"WheelRPM (RPM)":4900,"WheelTemperature (°C)":30,"CPUUsage (%)":35,"CPUTemperature (°C)":50,"SignalStrength (dBm)":-105,"GyroMagnitude (deg/s)":0.05,"Altitude (km)":503}},
    "borderline":    {"label":"Borderline / Mixed Signal","tier":"low","severity":"warning","desc":"Values near range edges across multiple subsystems — lower confidence expected.","values":{"OrbitPhase (%)":50,"Sunlight (0 or 1)":1,"BusVoltage (V)":25.5,"BusCurrent (A)":5.5,"BatteryVoltage (V)":3.3,"BatteryTemperature (°C)":42,"BatterySOC (%)":35,"SolarVoltage (V)":20,"SolarCurrent (A)":1.4,"WheelRPM (RPM)":6800,"WheelTemperature (°C)":60,"CPUUsage (%)":68,"CPUTemperature (°C)":78,"SignalStrength (dBm)":-95,"GyroMagnitude (deg/s)":1.2,"Altitude (km)":505}},
}
DEFAULT_TELEMETRY = dict(PRESET_SCENARIOS["nominal"]["values"])

# RUL thresholds — parameters to track for degradation
RUL_PARAMS = {
    "BatterySOC (%)":      {"threshold": 20, "direction": "down", "unit": "%"},
    "CPUTemperature (°C)": {"threshold": 85, "direction": "up",   "unit": "°C"},
    "BusVoltage (V)":      {"threshold": 26, "direction": "down", "unit": "V"},
}

TWIN = {"telemetry": dict(DEFAULT_TELEMETRY), "last": None, "history": [],
        "alerts": [], "rul_history": {k:[] for k in RUL_PARAMS}}

# ── Inference ─────────────────────────────────────────────────────────────
def infer(vals_dict, model_choice="xgb", include_lime=False):
    vals = [float(vals_dict.get(n, 0)) for n in feature_names]
    X    = np.array(vals, dtype=float).reshape(1, -1)

    if model_choice == "xgb":
        pred_enc  = int(xgb_model.predict(X)[0])
        cid       = int(xgb_le.inverse_transform([pred_enc])[0])
        proba_raw = xgb_model.predict_proba(X)[0]
        classes   = [int(c) for c in xgb_le.classes_]
        proba_map = {CLASS_LABELS[c]: round(float(p),6) for c,p in zip(classes,proba_raw)}
        conf      = float(proba_raw[list(xgb_le.classes_).index(cid)])

        # SHAP
        sv = np.array(shap_explainer.shap_values(X))
        pred_cls_idx = list(xgb_le.classes_).index(cid)
        shap_for_pred = sv[0, :, pred_cls_idx]
        shap_data = [{"name":n,"value":round(v,4),"shap_value":round(float(s),6),
                      "shap_abs":round(abs(float(s)),6),
                      "direction":"positive" if s>=0 else "negative"}
                     for n,v,s in zip(feature_names, vals, shap_for_pred)]
        shap_data.sort(key=lambda x: x["shap_abs"], reverse=True)
        model_used = "XGBoost"

    else:
        cid       = int(rf_model.predict(X)[0])
        proba_raw = rf_model.predict_proba(X)[0]
        classes   = [int(c) for c in rf_model.classes_]
        proba_map = {CLASS_LABELS[c]: round(float(p),6) for c,p in zip(classes,proba_raw)}
        conf      = float(proba_raw[classes.index(cid)])
        shap_data = [{"name":n,"value":round(v,4),
                      "shap_value":round(float(imp),6),
                      "shap_abs":round(float(imp),6),"direction":"positive"}
                     for n,v,imp in zip(feature_names, vals, rf_model.feature_importances_)]
        shap_data.sort(key=lambda x: x["shap_abs"], reverse=True)
        model_used = "RandomForest"

    # LIME (optional — slower, only on manual scans)
    lime_data = []
    if include_lime:
        try:
            # LIME needs predict_proba that returns shape (n_samples, n_all_classes=7)
            # XGBoost uses encoded labels (0-5 for 6 classes), so we must map back to
            # full 7-class space expected by LIME's class_names
            n_all_classes = 7

            def predict_fn(X_arr):
                X_arr = np.array(X_arr, dtype=float)
                if model_choice == "xgb":
                    raw = xgb_model.predict_proba(X_arr)  # shape (n, 6)
                    # Expand to full 7-class space; class 3 is missing from training data
                    full = np.zeros((raw.shape[0], n_all_classes), dtype=float)
                    for enc_idx, orig_cls in enumerate(xgb_le.classes_):
                        full[:, int(orig_cls)] = raw[:, enc_idx]
                    return full
                else:
                    raw = rf_model.predict_proba(X_arr)
                    full = np.zeros((raw.shape[0], n_all_classes), dtype=float)
                    for enc_idx, orig_cls in enumerate(rf_model.classes_):
                        full[:, int(orig_cls)] = raw[:, enc_idx]
                    return full

            # Use original class id (cid) as the LIME label since predict_fn
            # now returns full 7-class probabilities
            lime_exp = lime_explainer.explain_instance(
                np.array(vals, dtype=float),
                predict_fn,
                num_features=8,
                num_samples=300,
                labels=[cid],
            )
            for feat, weight in lime_exp.as_list(label=cid):
                lime_data.append({
                    "feature":   feat,
                    "weight":    round(float(weight), 6),
                    "direction": "positive" if weight >= 0 else "negative",
                })
            lime_data.sort(key=lambda x: abs(x["weight"]), reverse=True)

        except Exception as e:
            import traceback
            lime_data = [{"feature": f"LIME unavailable: {str(e)[:80]}",
                          "weight": 0, "direction": "positive"}]

    # Subsystem health
    sub_health = {}
    for sk, sv_meta in SUBSYSTEMS.items():
        oor = [f for f in sv_meta["features"]
               if f in NOMINAL_RANGES and vals_dict.get(f) is not None
               and (vals_dict[f] < NOMINAL_RANGES[f]["min"]
                    or vals_dict[f] > NOMINAL_RANGES[f]["max"])]
        sub_health[sk] = {"label":sv_meta["label"],"icon":sv_meta["icon"],
                          "status":"alert" if oor else "ok","out_of_range":oor}

    # RUL estimation
    rul = compute_rul(vals_dict)

    return {
        "class_id":         cid,
        "prediction":       CLASS_LABELS.get(cid, f"Class {cid}"),
        "severity":         CLASS_SEVERITY.get(cid, "warning"),
        "confidence":       round(conf, 6),
        "probabilities":    proba_map,
        "shap_values":      shap_data,
        "lime_values":      lime_data,
        "subsystem_health": sub_health,
        "recommendations":  CLASS_RECOMMENDATIONS.get(cid, []),
        "explanation":      CLASS_EXPLAIN.get(cid, ""),
        "model_used":       model_used,
        "rul":              rul,
        "timestamp":        datetime.now(timezone.utc).isoformat(),
        "telemetry":        {k: round(float(v),4) for k,v in vals_dict.items()},
    }

# ── RUL computation ───────────────────────────────────────────────────────
def compute_rul(vals_dict):
    rul_results = {}
    for param, cfg in RUL_PARAMS.items():
        val = vals_dict.get(param)
        if val is None:
            continue
        hist = TWIN["rul_history"][param]
        hist.append(float(val))
        if len(hist) > 30: hist.pop(0)

        threshold = cfg["threshold"]
        direction = cfg["direction"]
        unit      = cfg["unit"]

        if len(hist) < 3:
            rul_results[param] = {"status":"insufficient_data","rul_ticks":None,
                                  "current":round(float(val),2),"threshold":threshold,"unit":unit}
            continue

        # Linear regression on last N points to project when threshold is hit
        n = len(hist)
        x = np.arange(n, dtype=float)
        y = np.array(hist, dtype=float)
        slope = np.polyfit(x, y, 1)[0]

        if abs(slope) < 1e-6:
            rul_ticks = 9999
        elif direction == "down":
            rul_ticks = int((val - threshold) / (-slope)) if slope < 0 else 9999
        else:
            rul_ticks = int((threshold - val) / slope) if slope > 0 else 9999

        rul_ticks = max(0, min(rul_ticks, 9999))
        status = "critical" if rul_ticks < 5 else "warning" if rul_ticks < 20 else "nominal"

        rul_results[param] = {
            "status": status,
            "rul_ticks": rul_ticks,
            "slope": round(float(slope), 4),
            "current": round(float(val), 2),
            "threshold": threshold,
            "unit": unit,
        }
    return rul_results

def push_history(r):
    sev = r["severity"]
    TWIN["history"].append({"time":r["timestamp"],"prediction":r["prediction"],
                            "severity":sev,"confidence":r["confidence"],
                            "model":r.get("model_used","XGBoost")})
    if len(TWIN["history"]) > 300: TWIN["history"] = TWIN["history"][-300:]

    # Dynamic alert — push only for warning/critical and not duplicating last
    if sev in ("warning","critical"):
        last_alert = TWIN["alerts"][-1] if TWIN["alerts"] else None
        if not last_alert or last_alert["prediction"] != r["prediction"]:
            TWIN["alerts"].append({
                "time":      datetime.now(timezone.utc).isoformat(),
                "prediction":r["prediction"],
                "severity":  sev,
                "confidence":r["confidence"],
            })
            if len(TWIN["alerts"]) > 50: TWIN["alerts"] = TWIN["alerts"][-50:]

# ── Routes ────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html", features=feature_names)

@app.route("/predict", methods=["POST"])
def predict():
    p = request.get_json(force=True)
    if "features" not in p:
        return jsonify({"error":"Missing 'features'"}), 400
    if len(p["features"]) != len(feature_names):
        return jsonify({"error":f"Expected {len(feature_names)} features"}), 400
    try:
        vd  = {n:float(v) for n,v in zip(feature_names, p["features"])}
        mdl = p.get("model","xgb")
        inc_lime = p.get("lime", True)
        r   = infer(vd, mdl, include_lime=inc_lime)
        TWIN["telemetry"] = vd; TWIN["last"] = r
        push_history(r)
        return jsonify(r)
    except Exception as e:
        return jsonify({"error":str(e)}), 500

@app.route("/compare", methods=["POST"])
def compare():
    """Model comparison: run both XGBoost and RF on same input."""
    p = request.get_json(force=True)
    if "features" not in p:
        return jsonify({"error":"Missing 'features'"}), 400
    try:
        vd   = {n:float(v) for n,v in zip(feature_names, p["features"])}
        xgb_r = infer(vd, "xgb", include_lime=False)
        rf_r  = infer(vd, "rf",  include_lime=False)
        return jsonify({"xgb": xgb_r, "rf": rf_r})
    except Exception as e:
        return jsonify({"error":str(e)}), 500

@app.route("/simulate/tick", methods=["POST"])
def sim_tick():
    base = TWIN["telemetry"]
    nudged = {}
    for n in feature_names:
        rng   = NOMINAL_RANGES.get(n, {"min":0,"max":100})
        drift = (random.random()-0.5)*(rng["max"]-rng["min"])*0.025
        nudged[n] = round(base.get(n, DEFAULT_TELEMETRY.get(n,0)) + drift, 3)
    r = infer(nudged, "xgb", include_lime=False)
    TWIN["telemetry"] = nudged; TWIN["last"] = r
    push_history(r)
    return jsonify(r)

@app.route("/alerts",        methods=["GET"])
def get_alerts():   return jsonify({"alerts": list(reversed(TWIN["alerts"]))})
@app.route("/alerts/clear",  methods=["POST"])
def clr_alerts():   TWIN["alerts"]=[]; return jsonify({"status":"cleared"})
@app.route("/history",       methods=["GET"])
def get_history():  return jsonify({"history": TWIN["history"]})
@app.route("/history/clear", methods=["POST"])
def clr_history():  TWIN["history"]=[]; return jsonify({"status":"cleared"})
@app.route("/presets",       methods=["GET"])
def get_presets():  return jsonify(PRESET_SCENARIOS)
@app.route("/ranges",        methods=["GET"])
def get_ranges():   return jsonify(NOMINAL_RANGES)

@app.route("/subsystems", methods=["GET"])
def get_subsystems():
    sub_health = {}
    for sk, sv_meta in SUBSYSTEMS.items():
        oor=[f for f in sv_meta["features"]
             if f in NOMINAL_RANGES
             and (TWIN["telemetry"].get(f,0)<NOMINAL_RANGES[f]["min"]
                  or TWIN["telemetry"].get(f,0)>NOMINAL_RANGES[f]["max"])]
        sub_health[sk]={"label":sv_meta["label"],"icon":sv_meta["icon"],
                        "status":"alert" if oor else "ok","out_of_range":oor}
    return jsonify({"subsystems":sub_health,"last_prediction":TWIN["last"]})

@app.route("/analytics", methods=["GET"])
def analytics():
    h = TWIN["history"]
    if not h:
        return jsonify({"total":0,"by_class":{},"by_severity":{},"avg_confidence":0,"confidence_trend":[]})
    by_class,by_sev={},{}
    for row in h:
        by_class[row["prediction"]] = by_class.get(row["prediction"],0)+1
        by_sev[row["severity"]]     = by_sev.get(row["severity"],0)+1
    avg_conf = round(sum(r["confidence"] for r in h)/len(h),4)
    trend = [{"t":r["time"][:19].replace("T"," "),"c":round(r["confidence"]*100,2),"s":r["severity"]} for r in h[-50:]]
    return jsonify({"total":len(h),"by_class":by_class,"by_severity":by_sev,
                    "avg_confidence":avg_conf,"confidence_trend":trend})

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status":"operational","primary_model":"XGBoost",
                    "fallback":"RandomForest","xai":["SHAP (TreeExplainer)","LIME (TabularExplainer)"],
                    "features":len(feature_names),"classes":len(CLASS_LABELS),
                    "history":len(TWIN["history"]),"alerts":len(TWIN["alerts"])})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)