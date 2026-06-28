# Suspension Comfort Analyzer

A Streamlit dashboard for exploring a 2-DOF quarter-car suspension model. It simulates body and wheel response across road profiles, scores passenger comfort, shows frequency content, and can run a small tuning sweep over spring and damping values.

## Run

```powershell
pip install -r requirements.txt
streamlit run suspension_analyzer.py
```

## What it includes

- Real-time quarter-car simulation
- Smooth, wavy, rough, speed-bump, pothole, and random road inputs
- Comfort index based on RMS body acceleration thresholds
- Suspension travel and dissipated energy metrics
- Frequency spectrum with the 4-8 Hz human-sensitive band
- Grid-search tuning for spring stiffness and damping
- CSV export of simulated samples

