# SATMON — CubeSat Digital Twin

A full-stack fault detection system for CubeSat health monitoring. Submits satellite telemetry, classifies the fault type using XGBoost, and explains exactly which parameters drove that prediction using both SHAP and LIME. Built as a Flask web application with a live satellite simulator, model comparison, and remaining useful life estimation.

---

## What this actually does

You enter 16 telemetry values from a CubeSat — things like battery voltage, CPU temperature, signal strength, wheel RPM — and the system:

1. Runs an XGBoost classifier trained on satellite fault patterns
2. Tells you which of 7 fault classes the telemetry matches
3. Shows you *why* it made that decision, using two independent XAI methods
4. Estimates how long before key parameters hit critical thresholds
5. Suggests specific operator actions based on the detected fault

There's also a live simulator that continuously nudges the telemetry and re-runs the classifier, so you can watch the system respond in near real time without needing an actual satellite.

---

## Fault Classes

| ID | Class | Severity |
|----|-------|----------|
| 0 | Normal Operation | Nominal |
| 1 | Power Anomaly | Warning |
| 2 | Thermal Anomaly | Warning |
| 3 | Critical Fault | Critical |
| 4 | Communication Issue | Warning |
| 5 | Attitude Control Anomaly | Warning |
| 6 | CPU / System Overload | Warning |

---

## Screenshots

> Replace each placeholder below with a screenshot of the corresponding page.
> Recommended: run the app, load a preset from Quick Load, scan, then screenshot each section.

### Dashboard
![Dashboard](https://github.com/Sayeedahmed-09/SATMON/blob/2e651eaf5af4939bc0cd8a93553b424ccfe12309/docs/screenshots/dashboard(1).png)

The main overview page. Shows 4 live KPIs (mission state, confidence, subsystem alerts, session scans), a satellite banner, 9 module cards, a live telemetry chart that updates during simulation, and a snapshot of the last scan's raw telemetry values.

---

### Subsystem Health Monitor
![Subsystem Health](https://github.com/Sayeedahmed-09/SATMON/blob/c595039d38f1c612f64fd52b232f4340e501e34e/docs/screenshots/SubSystem%20Health(2).png)

Breaks the satellite into 6 subsystems — Power & Electrical, Thermal, Attitude Control, On-Board Computing, Communications, and Orbital. Each is shown as a circular health ring with a percentage score. If any parameter inside a subsystem is outside its nominal range, the ring turns red and lists the specific offending parameters.

---

### Analytics
![Analytics](https://github.com/Sayeedahmed-09/SATMON/blob/a39d5cad9220b832dd11d3962116227aad06fd51/docs/screenshots/Analytics(3).png)

Session-level statistics: total scans, average confidence, critical event count, and nominal rate. Includes a canvas-drawn confidence trend chart (last 50 scans, coloured dots per severity), a fault distribution bar chart, and a severity breakdown.

---

### Alerts
![Alerts](https://github.com/Sayeedahmed-09/SATMON/blob/b2ed91bd8ff0a2f3b7a0dd39c9786d82546595f2/docs/screenshots/Alerts(4).png)

Auto-generated alert log. Every time the classifier returns a warning or critical result, an entry is added here with a timestamp and confidence score. Consecutive identical alerts are deduplicated. Toast notifications also appear in the top-right corner of the screen.

---

### Manual Input
![Manual Input](https://github.com/Sayeedahmed-09/SATMON/blob/4bf3cf3db41dd1b5b1cbdc41786729abe1e1e223/docs/screenshots/Manual%20Input(5).png)

16 telemetry fields grouped by subsystem. Each field shows its nominal range below the input. Fields highlight amber if the entered value is outside the nominal range. You can choose between XGBoost (primary) and RandomForest (fallback) models, and toggle whether to include LIME analysis with the scan.

---

### Live Simulation
![Live Simulation](https://github.com/Sayeedahmed-09/SATMON/blob/20d7ff1965e443aafce05aa57aed171570d22e72/docs/screenshots/Live%20Simulation(6).png)

Starts a continuous simulation loop. Each tick nudges the current telemetry slightly (small random walk) and re-runs XGBoost + SHAP. The live feed shows each prediction as it arrives. Speed is adjustable from 0.5s to 2s per tick. The dashboard telemetry chart and subsystem rings update automatically.

---

### Quick Load
![Quick Load](https://github.com/Sayeedahmed-09/SATMON/blob/d5e95256d6992dce893138a1b941224a733ab7c7/docs/screenshots/Quick%20Load(7).png)

Six preset scenarios for fast testing — Normal Operation, Power Anomaly, Thermal Anomaly, Critical Fault, Communication Issue, and a Borderline/Mixed Signal scenario. Each card shows the expected severity and confidence level. "Load Only" fills the form so you can inspect and edit values first. "Load & Scan" runs immediately.

Also includes three randomize modes: High Confidence (values inside nominal range), Borderline (values near range edges), and Chaotic (multiple parameters outside nominal simultaneously).

---

### Classification Result
![Classification Result](https://github.com/Sayeedahmed-09/SATMON/blob/8afc143e4129670ff8084491f5a77cefa6c8e706/docs/screenshots/Classification(9).png)

The XGBoost output for the most recent scan. Shows the predicted class, severity badge, confidence percentage, and a probability bar chart for all 7 classes. The top class bar is highlighted. A model tag shows whether XGBoost or RandomForest was used.

---

### SHAP Analysis
![SHAP](https://github.com/Sayeedahmed-09/SATMON/blob/5f2ec2a53d0fed4efd6307b4e6724b3226fde138/docs/screenshots/SHAP(10).png)

SHAP (SHapley Additive Explanations) via TreeExplainer. For each of the top 10 features, shows how much it pushed the model toward or away from the predicted class — green for positive contribution, red for negative. Also shows the raw telemetry value that was submitted. Includes a plain-English narrative explaining what drove this specific prediction.

---

### LIME Analysis
![LIME]()

LIME (Local Interpretable Model-Agnostic Explanations). Works differently from SHAP — it perturbs the input 300 times, fits a local linear model around this specific prediction, and extracts which features mattered most locally. Available on manual scans only (not simulation ticks, because it's slower). Comparing SHAP and LIME outputs gives a second opinion on which parameters were most influential.

---

### Model Comparison
![Model Comparison](https://github.com/Sayeedahmed-09/SATMON/blob/01c074ade65ee0186abb209ff577e5a3793f2fcf/docs/screenshots/Model%20Comparision(12).png)

Runs both XGBoost and RandomForest on the exact same telemetry and shows the results side by side — predictions, confidence, probability bars, and top SHAP features for each model. A verdict at the bottom shows whether the models agree or disagree, which is useful for assessing prediction reliability.

---

### Remaining Useful Life
![RUL](https://github.com/Sayeedahmed-09/SATMON/blob/8aa9db83c386e6bfb1d0dec4db099d65289c3ed5/docs/screenshots/Remaining%20Life(13).png)

Tracks Battery SOC, CPU Temperature, and Bus Voltage over recent scans. Fits a linear regression to each parameter's recent history and projects how many more ticks before it crosses its critical threshold. Shows current value, threshold, trend slope, and a status bar. Requires at least 3 data points — run a few scans or start the simulator.

---

### Recommendations
![Recommendations](https://github.com/Sayeedahmed-09/SATMON/blob/deb99913844a184e12fcfa34dca0993a31440fa6/docs/screenshots/Recommandations(14).png)

A list of specific operator actions based on the detected fault class. Not generic — each fault class has its own set of 3–4 actionable recommendations. For example, a Communication Issue suggests re-pointing the antenna, switching to backup frequency, and logging the outage duration.

---

### Mission Timeline
![Timeline](https://github.com/Sayeedahmed-09/SATMON/blob/0333def1f7610586109c3ca2de557e9e649ba40e/docs/screenshots/Mission%20Timeline(15).png)

Visual event log of all predictions this session. Each prediction appears as a coloured dot on a vertical timeline with the fault class, confidence, and timestamp. Colour-coded by severity — green for nominal, amber for warning, red for critical.

---

### Prediction History
![History](https://github.com/Sayeedahmed-09/SATMON/blob/af130d8b1ab811c32ce2fe73aa158a2de97347f2/docs/screenshots/Prediction%20History(16).png)

Full table of all scans this session — timestamp, prediction, confidence, and which model was used. Sortable by most recent. Clearable.

---

### Export
![Export](https://github.com/Sayeedahmed-09/SATMON/blob/0c3768541d63f6e838a62e1d3f843eb9e956181c/docs/screenshots/Export(17).png)

Four export options:
- **Export JSON** — full prediction payload including SHAP values, LIME weights, subsystem health, recommendations, and raw telemetry
- **Telemetry CSV** — live chart buffer exported as a CSV with timestamps
- **Copy Summary** — copies a plain-text summary with top SHAP features to clipboard
- **Print / PDF** — browser print dialog with a print-optimised layout (sidebar and controls hidden)

---

## Tech Stack

**Backend**
- Python 3.x
- Flask
- XGBoost (primary classifier)
- scikit-learn RandomForest (fallback / comparison)
- SHAP — TreeExplainer for feature attribution
- LIME — LimeTabularExplainer for local explanations
- NumPy, Joblib

**Frontend**
- Vanilla HTML + CSS + JavaScript (no frameworks)
- Inter + JetBrains Mono fonts
- Canvas API for live telemetry chart and confidence trend
- SVG for subsystem health rings and satellite illustration

---

## Project Structure

```
satmon/
│
├── app.py                        # Flask application — all routes and inference logic
├── train_xgb.py                  # XGBoost training script (run once after cloning)
│
├── satellite_model.pkl           # RandomForest baseline model
├── feature_names.pkl             # List of 16 telemetry feature names
├── xgb_model.joblib              # XGBoost primary model
├── xgb_label_encoder.joblib      # Label encoder for XGBoost class mapping
│
├── templates/
│   └── index.html                # Single-page app — all 17 views in one file
│
├── static/
│   ├── style.css                 # Complete design system (973 lines)
│   └── main.js                   # All frontend logic — nav, rendering, chart, export
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Installation

**Clone the repo**
```bash
git clone https://github.com/YOUR_USERNAME/satmon.git
cd satmon
```

**Create a virtual environment (recommended)**
```bash
python -m venv myenv
myenv\Scripts\activate        # Windows
# source myenv/bin/activate   # macOS / Linux
```

**Install dependencies**
```bash
pip install -r requirements.txt
```

**Retrain the XGBoost model locally**

The `.joblib` files are machine-specific. Run this once after cloning:
```bash
python train_xgb.py
```
This takes about 30–60 seconds and generates `xgb_model.joblib` and `xgb_label_encoder.joblib` in the project folder.

**Run the application**
```bash
python app.py
```

Open `http://localhost:5000` in your browser.

---

## Quick Start

1. Go to **Quick Load** in the sidebar
2. Click **Load & Scan** on any preset (e.g. Critical Fault)
3. The app jumps to **Classification** — check the predicted class and confidence
4. Go to **SHAP** — see which parameters drove the prediction
5. Go to **LIME** — compare with the local explanation
6. Go to **Model Comparison** — click Run Comparison to see XGBoost vs RandomForest
7. Go to **Live Simulation** — click Start and watch the dashboard update in real time
8. After a few ticks, go to **Remaining Life** to see RUL projections

---

## Model Details

The XGBoost model was trained via knowledge distillation from the RandomForest baseline — 20,000 synthetic telemetry samples were generated across the full parameter range, labelled by the RF, and used to train XGBoost. This approach transfers the RF's learned decision boundary to XGBoost, which is required for SHAP's TreeExplainer and gives faster inference.

| Model | Type | Training samples | Classes |
|-------|------|-----------------|---------|
| RandomForest | Baseline | Original dataset | 7 |
| XGBoost | Primary (distilled) | 20,000 synthetic | 6 active |

The LIME explainer uses 300 perturbations per prediction and maps XGBoost's encoded class labels back to the full 7-class probability space before computing local feature weights.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Serves the dashboard |
| POST | `/predict` | Run XGBoost or RF inference + SHAP + LIME |
| POST | `/simulate/tick` | One simulation tick (XGBoost only, no LIME) |
| POST | `/compare` | Run both models on same input |
| GET | `/presets` | Preset telemetry scenarios |
| GET | `/ranges` | Nominal min/max per feature |
| GET | `/subsystems` | Subsystem health based on last telemetry |
| GET | `/analytics` | Session stats + confidence trend |
| GET | `/alerts` | Alert log |
| POST | `/alerts/clear` | Clear alert log |
| GET | `/history` | Prediction history |
| POST | `/history/clear` | Clear history |
| GET | `/health` | System status |

---

## requirements.txt

```
flask
scikit-learn
xgboost
shap
lime
numpy
joblib
```

---

*Built as part of a CubeSat health monitoring research project. The satellite simulator and fault classifier are designed for educational and demonstration purposes.*
