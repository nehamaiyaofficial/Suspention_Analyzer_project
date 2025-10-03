import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from collections import deque
from sklearn.linear_model import LinearRegression

# --- Page & CSS Configuration ---
st.set_page_config(page_title="Suspension Comfort Analyzer PRO", page_icon="🚗", layout="wide")
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
    }
    .stMetric {
        st.markdown(""
        min-width: 180px !important;  /* Enforce minimum width */
        padding: 1.2rem !important;   /* Uniform padding */
        border-radius: 10px !important;
        font-size: 1.15rem !important;
        box-sizing: border-box !important;
  

        background: linear-gradient(135deg, rgba(59, 130, 246, 0.1), rgba(37, 99, 235, 0.05));
        padding: 1.1rem;
        border-radius: 10px;
        border: 1px solid rgba(59, 130, 246, 0.3);
    }
    .warning {
        color: #f97316;
        font-weight: bold;
    }
    .alert {
        color: #ef4444;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# --- Initialize session state ---
if 'initialized' not in st.session_state:
    st.session_state.time = 0.0
    st.session_state.z1 = 0.0
    st.session_state.v1 = 0.0
    st.session_state.z2 = 0.0
    st.session_state.v2 = 0.0
    st.session_state.accel_history = deque(maxlen=600)
    st.session_state.chart_data = deque(maxlen=600)
    st.session_state.suspension_travel_history = deque(maxlen=600)
    st.session_state.is_running = False
    st.session_state.total_distance = 0.0
    st.session_state.power_consumption = 0.0
    st.session_state.simulation_runs = []
    st.session_state.initialized = True

# --- Road profile with amplitude control ---
def get_road_excitation(t, profile, velocity, amplitude=1.0):
    v = velocity / 3.6
    if profile == 'Smooth Road':
        return amplitude * 0.01 * np.sin(2 * np.pi * 0.5 * t)
    elif profile == 'Wavy Road':
        return amplitude * (0.03 * np.sin(2 * np.pi * 1.2 * t) + 0.015 * np.sin(2 * np.pi * 3 * t))
    elif profile == 'Rough Road':
        return amplitude * (0.05 * np.sin(2 * np.pi * 2 * t) + 0.025 * np.sin(2 * np.pi * 5 * t) + 0.02 * np.sin(2 * np.pi * 8 * t))
    elif profile == 'Speed Bumps':
        bump_pos = int(t / 2.0) * 2.0 + 1
        dist = abs(t - bump_pos)
        return amplitude * (0.08 * np.exp(-25 * dist * dist) if dist < 0.2 else 0)
    elif profile == 'Pothole':
        pothole_pos = int(t / 4.0) * 4.0 + 2
        dist = abs(t - pothole_pos)
        return amplitude * (-0.06 * np.exp(-30 * dist * dist) if dist < 0.15 else 0)
    elif profile == 'Random Road':
        np.random.seed(int(t * 100))
        return amplitude * 0.03 * (np.random.randn() * 0.5 + 0.02 * np.sin(2 * np.pi * 2 * t))
    return 0

# --- ISO 2631-1 comfort calculations ---
def calculate_comfort_index(accel_list):
    if len(accel_list) == 0:
        return 100.0
    accel_array = np.array(accel_list)
    rms = np.sqrt(np.mean(accel_array**2))
    if rms < 0.315:
        index = 100 - (rms / 0.315) * 10
    elif rms < 0.63:
        index = 90 - ((rms - 0.315) / 0.315) * 30
    elif rms < 1.0:
        index = 60 - ((rms - 0.63) / 0.37) * 30
    elif rms < 2.0:
        index = 30 - ((rms - 1.0) / 1.0) * 20
    else:
        index = max(0, 10 - ((rms - 2.0) / 2.0) * 10)
    return max(0.0, min(100.0, index))

def get_comfort_rating(index):
    if index >= 90: return 'Excellent', '🟢', '#10b981'
    elif index >= 70: return 'Good', '🔵', '#3b82f6'
    elif index >= 50: return 'Fair', '🟡', '#fbbf24'
    elif index >= 30: return 'Poor', '🟠', '#f97316'
    else: return 'Very Poor', '🔴', '#ef4444'

def calculate_frequency_response(accel_history, dt=0.01):
    if len(accel_history) < 50:
        return [], []
    accel_array = np.array(list(accel_history))
    n = len(accel_array)
    fft_vals = np.fft.fft(accel_array)
    fft_freq = np.fft.fftfreq(n, dt)
    pos_mask = fft_freq > 0
    freqs = fft_freq[pos_mask]
    mag = np.abs(fft_vals[pos_mask]) / n
    freq_mask = freqs <= 20
    return freqs[freq_mask], mag[freq_mask]

# --- Suspension dynamics using Euler integration ---
def update_suspension_dynamics(dt, params, road_profile, road_amplitude):
    m1 = params['sprung_mass']
    m2 = params['unsprung_mass']
    k1 = params['spring_stiffness']
    c1 = params['damping_coeff']
    k2 = params['tire_stiffness']
    c2 = params['tire_damping']
    road_input = get_road_excitation(st.session_state.time, road_profile, params['velocity'], road_amplitude)
    suspension_force = k1 * (st.session_state.z2 - st.session_state.z1) + c1 * (st.session_state.v2 - st.session_state.v1)
    tire_force = k2 * (road_input - st.session_state.z2) + c2 * (-st.session_state.v2)
    a1 = suspension_force / m1
    a2 = (tire_force - suspension_force) / m2
    st.session_state.v1 += a1 * dt
    st.session_state.z1 += st.session_state.v1 * dt
    st.session_state.v2 += a2 * dt
    st.session_state.z2 += st.session_state.v2 * dt
    st.session_state.accel_history.append(abs(a1))
    suspension_travel = abs(st.session_state.z2 - st.session_state.z1)
    st.session_state.suspension_travel_history.append(suspension_travel)
    relative_velocity = abs(st.session_state.v2 - st.session_state.v1)
    power = c1 * relative_velocity**2
    st.session_state.power_consumption += power * dt
    st.session_state.total_distance += (params['velocity'] / 3.6) * dt
    return a1, a2, road_input, suspension_travel

def reset_simulation():
    st.session_state.time = 0.0
    st.session_state.z1 = 0.0
    st.session_state.v1 = 0.0
    st.session_state.z2 = 0.0
    st.session_state.v2 = 0.0
    st.session_state.accel_history.clear()
    st.session_state.chart_data.clear()
    st.session_state.suspension_travel_history.clear()
    st.session_state.is_running = False
    st.session_state.total_distance = 0.0
    st.session_state.power_consumption = 0.0

# --- Optimization function (grid search on spring and damper) ---
def grid_search_optimization(params, param_ranges, road_profile, road_amplitude, dt=0.01, steps=30):
    best_score = -np.inf
    best_params = None
    results = []
    for spring_k in param_ranges['spring_stiffness']:
        for damping_c in param_ranges['damping_coeff']:
            # Prepare simulation params
            test_params = params.copy()
            test_params['spring_stiffness'] = spring_k
            test_params['damping_coeff'] = damping_c
            # Reset simulation state
            st.session_state.time = 0.0
            st.session_state.z1 = 0.0
            st.session_state.v1 = 0.0
            st.session_state.z2 = 0.0
            st.session_state.v2 = 0.0
            accel_local = []
            for _ in range(steps):
                m1 = test_params['sprung_mass']
                m2 = test_params['unsprung_mass']
                k1 = test_params['spring_stiffness']
                c1 = test_params['damping_coeff']
                k2 = test_params['tire_stiffness']
                c2 = test_params['tire_damping']
                t = st.session_state.time
                road_input = get_road_excitation(t, road_profile, test_params['velocity'], road_amplitude)
                suspension_force = k1 * (st.session_state.z2 - st.session_state.z1) + c1 * (st.session_state.v2 - st.session_state.v1)
                tire_force = k2 * (road_input - st.session_state.z2) + c2 * (-st.session_state.v2)
                a1 = suspension_force / m1
                a2 = (tire_force - suspension_force) / m2
                st.session_state.v1 += a1 * dt
                st.session_state.z1 += st.session_state.v1 * dt
                st.session_state.v2 += a2 * dt
                st.session_state.z2 += st.session_state.v2 * dt
                accel_local.append(abs(a1))
                st.session_state.time += dt
            comfort_idx = calculate_comfort_index(accel_local)
            results.append({'spring_stiffness': spring_k, 'damping_coeff': damping_c, 'comfort_index': comfort_idx})
            if comfort_idx > best_score:
                best_score = comfort_idx
                best_params = (spring_k, damping_c)
    return best_params, best_score, results

# --- Main UI ---

# Title & description
st.markdown("""
    <h1 style='text-align:center; background: linear-gradient(90deg, #3b82f6, #8b5cf6, #ec4899);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 3em;'>
    Suspension Comfort Analyzer PRO
    </h1>
    <p style='text-align: center; color: #94a3b8; font-size: 1.15em;'>
    Real-time Quarter-Car Model • ISO 2631-1 Comfort • Frequency Analysis & Optimization
    </p>
""", unsafe_allow_html=True)

# Sidebar: parameters
st.sidebar.header("⚙️ Suspension Parameters")
sb1, sb2 = st.sidebar.columns(2)
with sb1:
    sprung_mass = st.slider("Sprung Mass (kg)", 250, 1600, 1400, 25)  # increased max to 1600 to fit default 1400
    spring_stiffness = st.slider("Spring Stiffness (kN/m)", 10, 1600, 1400, 5) * 1000
with sb2:
    damping_coeff = st.slider("Damping (Ns/m)", 500, 15000, 8000, 250)  # increased max and default
    velocity = st.slider("Velocity (km/h)", 20, 200, 120, 10)  # increased max and default

st.sidebar.header("🛣️ Road Conditions")
road_profile = st.sidebar.selectbox(
    "Road Profile:", ["Smooth Road", "Wavy Road", "Rough Road", "Speed Bumps", "Pothole", "Random Road"])
road_amplitude = st.sidebar.slider("Road Severity", 0.5, 2.0, 1.0, 0.1)

road_descriptions = {
    "Smooth Road": "🛣️ Highway – minimal surface input.",
    "Wavy Road": "〰️ Gentle and periodic undulations.",
    "Rough Road": "🌊 Poor surface, intense high frequency.",
    "Speed Bumps": "⚠️ Discrete, repeat obstacles.",
    "Pothole": "🕳️ Single, sharp depression.",
    "Random Road": "🎲 Simulated stochastic noise."
}
st.sidebar.info(road_descriptions[road_profile])

with st.sidebar.expander("🔧 Advanced Settings"):
    unsprung_mass = st.slider("Unsprung Mass (kg)", 50, 250, 100, 10)
    tire_stiffness = st.slider("Tire Stiffness (kN/m)", 200, 1000, 400, 50) * 1000
    tire_damping = st.slider("Tire Damping (Ns/m)", 500, 3500, 1000, 100)
    show_frequency = st.checkbox("Show Frequency Analysis", value=True)
    show_suspension_travel = st.checkbox("Show Suspension Travel", value=True)
    show_optimization = st.checkbox("Enable Optimization Tool", value=True)

params = {
    'sprung_mass': sprung_mass, 'unsprung_mass': unsprung_mass,
    'spring_stiffness': spring_stiffness, 'damping_coeff': damping_coeff,
    'tire_stiffness': tire_stiffness, 'tire_damping': tire_damping, 'velocity': velocity
}

# Controls
st.markdown("---")
colA, colB, colC, colD = st.columns([1, 1, 3, 3])

with colA:
    if st.button("▶️ Start" if not st.session_state.is_running else "⏸️ Pause", use_container_width=True, type="primary"):
        st.session_state.is_running = not st.session_state.is_running
        st.rerun()

with colB:
    if st.button("🔄 Reset", use_container_width=True):
        reset_simulation()
        st.rerun()

with colC:
    if st.button("📊 Export CSV", use_container_width=True) and len(st.session_state.chart_data) > 0:
        df = pd.DataFrame(list(st.session_state.chart_data))
        st.download_button("⬇️ Download Data", df.to_csv(index=False), "suspension_data.csv", "text/csv")

# Optimization button and process
opt_result = None
if show_optimization:
    with colD:
        if st.button("🚀 Optimize for Comfort", use_container_width=True):
            st.info("Running optimization... This may take some time.")
            spring_range = np.arange(10_000, 80_001, 7000)
            damping_range = np.arange(500, 8001, 700)
            best_params, best_score, results = grid_search_optimization(
                params, {'spring_stiffness': spring_range, 'damping_coeff': damping_range}, road_profile, road_amplitude
            )
            st.session_state.simulation_runs.append({
                'params': params.copy(), 'results': results,
                'best_spring': best_params[0], 'best_damping': best_params[1], 'best_score': best_score
            })
            opt_result = (best_params, best_score)
            st.success(f"Optimization Complete! Best Comfort: {best_score:.1f} at Spring {best_params[0]/1000:.1f} kN/m and Damping {best_params[1]:.0f} Ns/m")

# Simulation loop
dt = 0.01
max_iter = 12
if st.session_state.is_running:
    for _ in range(max_iter):
        a1, a2, road_input, susp_travel = update_suspension_dynamics(
            dt, params, road_profile, road_amplitude
        )
        st.session_state.time += dt
        st.session_state.chart_data.append({
            'time': st.session_state.time,
            'body_accel': a1,
            'wheel_accel': a2,
            'road_input': road_input * 100,
            'body_disp': st.session_state.z1 * 100,
            'suspension_travel': susp_travel * 100
        })
    time.sleep(0.03)
    st.stop()

# Metrics calculation
if len(st.session_state.accel_history) > 0:
    accel_array = np.array(st.session_state.accel_history)
    rms_accel = np.sqrt(np.mean(accel_array ** 2))
    peak_accel = np.max(accel_array)
    vdv = np.power(np.mean(np.power(np.abs(accel_array), 4)), 0.25)
    comfort_index = calculate_comfort_index(st.session_state.accel_history)
    avg_susp_travel = np.mean(st.session_state.suspension_travel_history) * 100
    max_susp_travel = np.max(st.session_state.suspension_travel_history) * 100
else:
    rms_accel = peak_accel = vdv = avg_susp_travel = max_susp_travel = 0.0
    comfort_index = 100.0

comfort_text, comfort_emoji, comfort_color = get_comfort_rating(comfort_index)

# Display core metrics
st.markdown("---")
mcols = st.columns(5)
with mcols[0]:
    st.metric(f"{comfort_emoji} Comfort Index", f"{comfort_index:.1f}/100")
    st.markdown(
        f"<span style='display:inline-block; background:{comfort_color}; color:white; "
        f"padding:0.5em 1.1em; border-radius:1em; font-weight:600; font-size:1em;'>{comfort_text}</span>",
        unsafe_allow_html=True
    )
mcols[1].metric("📊 RMS Accel", f"{rms_accel:.3f} m/s²", delta="Low" if rms_accel < 0.5 else "High", delta_color="normal" if rms_accel < 0.5 else "inverse")
mcols[2].metric("⚡ Peak Accel", f"{peak_accel:.3f} m/s²")
mcols[3].metric("📏 Suspension Travel", f"{avg_susp_travel:.1f} mm", delta=f"Max {max_susp_travel:.1f} mm")
mcols[4].metric("🔋 Dissipated Energy", f"{st.session_state.power_consumption/1000:.1f} kJ", delta=f"{st.session_state.total_distance:.0f} m")

# Alerts on system conditions
st.markdown("---")
damping_ratio = damping_coeff / (2 * np.sqrt(spring_stiffness * sprung_mass))
if damping_ratio < 0.3:
    st.warning("⚠️ Under-damped suspension: Consider increasing damping for improved comfort.")
elif damping_ratio > 0.7:
    st.info("ℹ️ Over-damped suspension: Comfort may decrease; consider reducing damping.")
if max_susp_travel > 8:
    st.error(f"❌ High suspension travel ({max_susp_travel:.1f} mm): Risk of bottoming out.")
if rms_accel > 1.0:
    st.error("❌ High vibrations detected: Poor comfort.")

# Main response charts
st.markdown("---")
st.subheader("📈 Dynamic Response Analysis")
if len(st.session_state.chart_data) > 0:
    df = pd.DataFrame(list(st.session_state.chart_data))
    fig = make_subplots(
        rows=2 if show_suspension_travel else 1, cols=1,
        subplot_titles=('Acceleration Response', 'Suspension Travel') if show_suspension_travel else ('Acceleration Response',),
        vertical_spacing=0.13, row_heights=[0.6, 0.4] if show_suspension_travel else [1.0]
    )
    fig.add_trace(go.Scatter(x=df['time'], y=df['body_accel'], mode='lines', name='Body Accel',
                             line=dict(color='#3b82f6', width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['time'], y=df['wheel_accel'], mode='lines', name='Wheel Accel',
                             line=dict(color='#f59e0b', width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['time'], y=df['road_input'] / 10, mode='lines', name='Road Input (×0.1)',
                             line=dict(color='#6b7280', width=1, dash='dash'), opacity=0.5), row=1, col=1)
    if show_suspension_travel:
        fig.add_trace(go.Scatter(x=df['time'], y=df['suspension_travel'], mode='lines', name='Suspension Travel',
                                 line=dict(color='#8b5cf6', width=2), fill='tozeroy', fillcolor='rgba(139, 92, 246, 0.15)'),
                      row=2, col=1)
        fig.update_yaxes(title_text="Travel (cm)", row=2, col=1)
    fig.update_xaxes(title_text="Time (s)", row=2 if show_suspension_travel else 1, col=1)
    fig.update_yaxes(title_text="Acceleration (m/s²)", row=1, col=1)
    fig.update_layout(
        height=600 if show_suspension_travel else 400,
        template="plotly_dark",
        hovermode='x unified',
        showlegend=True,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15,23,42,0.5)',
        legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("⏯️ Press **Start** to begin simulation.")

# Frequency analysis
if show_frequency and len(st.session_state.accel_history) >= 50:
    colf1, colf2 = st.columns([2, 1])
    with colf1:
        st.subheader("🎵 Frequency Spectrum Analysis")
        freqs, magnitude = calculate_frequency_response(st.session_state.accel_history)
        fig_freq = go.Figure()
        fig_freq.add_trace(go.Scatter(x=freqs, y=magnitude, mode='lines',
                                      line=dict(color='#ec4899', width=2), fill='tozeroy', fillcolor='rgba(236,72,153,0.3)',
                                      name='Magnitude'))
        fig_freq.add_vrect(x0=4, x1=8, fillcolor="rgba(251,191,36,0.11)", layer="below", line_width=0,
                           annotation_text="Human Sensitive Band (4-8 Hz)", annotation_position="top left")
        fig_freq.update_layout(height=300, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(15,23,42,0.5)',
                               xaxis_title="Frequency (Hz)", yaxis_title="Magnitude")
        st.plotly_chart(fig_freq, use_container_width=True)
    with colf2:
        st.subheader("📊 Frequency Statistics")
        if len(freqs) > 0:
            dom_idx = np.argmax(magnitude)
            dom_freq = freqs[dom_idx]
            st.metric("Dominant Frequency", f"{dom_freq:.2f} Hz")
            nat_freq = np.sqrt(spring_stiffness / sprung_mass) / (2 * np.pi)
            st.metric("Natural Frequency", f"{nat_freq:.2f} Hz")
            damping_ratio = damping_coeff / (2 * np.sqrt(spring_stiffness * sprung_mass))
            st.metric("Damping Ratio", f"{damping_ratio:.3f}", delta="Optimal" if 0.3 <= damping_ratio <= 0.7 else "Sub-optimal")
        else:
            st.info("Insufficient data for frequency statistics.")

# Simple ML prediction for RMS acceleration (using historical runs if available)
st.markdown("---")
if len(st.session_state.simulation_runs) >= 5:
    history_df = pd.DataFrame()
    for run in st.session_state.simulation_runs:
        res = run['results']
        df_run = pd.DataFrame(res)
        df_run['sprung_mass'] = run['params']['sprung_mass']
        df_run['unsprung_mass'] = run['params']['unsprung_mass']
        df_run['velocity'] = run['params']['velocity']
        history_df = pd.concat([history_df, df_run], ignore_index=True)

    if not history_df.empty:
        try:
            X = history_df[['spring_stiffness', 'damping_coeff', 'velocity']]
            y = history_df['comfort_index']
            model = LinearRegression().fit(X, y)
            pred_comfort = model.predict([[spring_stiffness, damping_coeff, velocity]])[0]
            st.metric("🤖 ML Predicted Comfort Index", f"{pred_comfort:.1f}/100", delta="Based on simulation history")
        except Exception:
            pass

# Technical info and formulas
st.markdown("---")
colt1, colt2 = st.columns(2)
with colt1:
    st.subheader("📐 Quarter-Car Model Equations")
    st.code(
        f"""Sprung Mass (Body):
m₁·z̈₁ = k₁(z₂ - z₁) + c₁(ż₂ - ż₁)
Unsprung Mass (Wheel):
m₂·z̈₂ = k₂(zᵣ - z₂) + c₂(żᵣ - ż₂) - k₁(z₂ - z₁) - c₁(ż₂ - ż₁)
Parameters:
m₁ = {sprung_mass} kg (sprung mass)
m₂ = {unsprung_mass} kg (unsprung mass)
k₁ = {spring_stiffness/1000:.0f} kN/m
c₁ = {damping_coeff:.0f} Ns/m
k₂ = {tire_stiffness/1000:.0f} kN/m
c₂ = {tire_damping:.0f} Ns/m
""", language='text')

with colt2:
    st.subheader("🎯 ISO 2631-1 Comfort Standard")
    st.markdown(f"""
    <b>Current Comfort Level:</b> <span style='color:{comfort_color}; font-weight:bold; font-size:1.2em;'>{comfort_text}</span><br>
    **RMS Acceleration:** {rms_accel:.3f} m/s²<br>
    **VDV:** {vdv:.3f} m/s¹·⁷⁵<br>
    <br>
    <b>Comfort Thresholds:</b><br>
    - 🟢 < 0.315 m/s²: Not uncomfortable<br>
    - 🔵 0.315 - 0.63 m/s²: Slightly uncomfortable<br>
    - 🟡 0.63 - 1.0 m/s²: Fairly uncomfortable<br>
    - 🟠 1.0 - 2.0 m/s²: Uncomfortable<br>
    - 🔴 > 2.0 m/s²: Very uncomfortable<br>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; color: #64748b; padding: 20px; font-size: 0.9em;'>
    🚗 <b>Model:</b> 2-DOF Quarter-Car &bull; <b>Standard:</b> ISO 2631-1 &bull; <b>Integration:</b> Euler (Δt=10ms)<br>
    ⏱️ Time: {st.session_state.time:.2f}s &bull; 📊 Data Points: {len(st.session_state.chart_data)} &bull; 
    🏃 Status: {'✅ Running' if st.session_state.is_running else '⏸️ Paused'} &bull;
    🛣️ Distance: {st.session_state.total_distance:.1f} m
</div>
""", unsafe_allow_html=True)
