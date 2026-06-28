from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots


MAX_HISTORY = 600
DT_SECONDS = 0.01
STEPS_PER_REFRESH = 12


@dataclass
class VehicleParams:
    sprung_mass: float
    unsprung_mass: float
    spring_stiffness: float
    damping_coeff: float
    tire_stiffness: float
    tire_damping: float
    velocity: float


@dataclass
class QuarterCarState:
    time: float = 0.0
    z_body: float = 0.0
    v_body: float = 0.0
    z_wheel: float = 0.0
    v_wheel: float = 0.0


def configure_page() -> None:
    st.set_page_config(
        page_title="Suspension Comfort Analyzer",
        page_icon=":car:",
        layout="wide",
    )
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at 15% 10%, rgba(20, 184, 166, 0.14), transparent 30%),
                radial-gradient(circle at 85% 5%, rgba(249, 115, 22, 0.10), transparent 28%),
                linear-gradient(135deg, #0b1120 0%, #162033 48%, #101827 100%);
            color: #e5edf7;
        }
        section[data-testid="stSidebar"] {
            background: rgba(8, 13, 25, 0.88);
            border-right: 1px solid rgba(148, 163, 184, 0.16);
        }
        div[data-testid="stMetric"] {
            min-height: 116px;
            padding: 1rem;
            border-radius: 8px;
            border: 1px solid rgba(148, 163, 184, 0.20);
            background: rgba(15, 23, 42, 0.72);
            box-shadow: 0 10px 24px rgba(0, 0, 0, 0.18);
        }
        div[data-testid="stMetricValue"] {
            font-size: 1.55rem;
        }
        .comfort-pill {
            display: inline-block;
            margin-top: 0.55rem;
            padding: 0.4rem 0.8rem;
            border-radius: 999px;
            color: #ffffff;
            font-weight: 700;
            font-size: 0.9rem;
        }
        .app-subtitle {
            color: #a9b7cc;
            text-align: center;
            font-size: 1.05rem;
            margin-bottom: 0.4rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def initialize_state() -> None:
    defaults = {
        "model_state": QuarterCarState(),
        "accel_history": deque(maxlen=MAX_HISTORY),
        "chart_data": deque(maxlen=MAX_HISTORY),
        "travel_history": deque(maxlen=MAX_HISTORY),
        "is_running": False,
        "total_distance": 0.0,
        "dissipated_energy": 0.0,
        "optimization_runs": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_simulation() -> None:
    st.session_state.model_state = QuarterCarState()
    st.session_state.accel_history.clear()
    st.session_state.chart_data.clear()
    st.session_state.travel_history.clear()
    st.session_state.is_running = False
    st.session_state.total_distance = 0.0
    st.session_state.dissipated_energy = 0.0


def get_road_excitation(
    t: float,
    profile: str,
    velocity_kmh: float,
    amplitude: float = 1.0,
) -> float:
    speed_ms = velocity_kmh / 3.6

    if profile == "Smooth Road":
        return amplitude * 0.01 * np.sin(2 * np.pi * 0.5 * t)
    if profile == "Wavy Road":
        return amplitude * (
            0.026 * np.sin(2 * np.pi * (0.65 + speed_ms / 80) * t)
            + 0.014 * np.sin(2 * np.pi * 2.4 * t)
        )
    if profile == "Rough Road":
        return amplitude * (
            0.035 * np.sin(2 * np.pi * 1.8 * t)
            + 0.020 * np.sin(2 * np.pi * 5.2 * t)
            + 0.014 * np.sin(2 * np.pi * 8.5 * t)
        )
    if profile == "Speed Bumps":
        bump_spacing_s = max(0.9, 36 / max(speed_ms, 0.1))
        bump_center = round(t / bump_spacing_s) * bump_spacing_s
        distance = abs(t - bump_center)
        return amplitude * (0.075 * np.exp(-22 * distance * distance) if distance < 0.35 else 0.0)
    if profile == "Pothole":
        pothole_spacing_s = max(1.6, 60 / max(speed_ms, 0.1))
        pothole_center = round((t - 0.8) / pothole_spacing_s) * pothole_spacing_s + 0.8
        distance = abs(t - pothole_center)
        return amplitude * (-0.055 * np.exp(-32 * distance * distance) if distance < 0.28 else 0.0)
    if profile == "Random Road":
        rng = np.random.default_rng(int(t * 120))
        noise = rng.normal(0.0, 0.012)
        return amplitude * (noise + 0.018 * np.sin(2 * np.pi * 2.1 * t))

    return 0.0


def simulate_step(
    state: QuarterCarState,
    params: VehicleParams,
    road_profile: str,
    road_amplitude: float,
    dt: float = DT_SECONDS,
) -> tuple[QuarterCarState, dict[str, float]]:
    road_input = get_road_excitation(
        state.time,
        road_profile,
        params.velocity,
        road_amplitude,
    )
    suspension_force = (
        params.spring_stiffness * (state.z_wheel - state.z_body)
        + params.damping_coeff * (state.v_wheel - state.v_body)
    )
    tire_force = (
        params.tire_stiffness * (road_input - state.z_wheel)
        - params.tire_damping * state.v_wheel
    )

    body_accel = suspension_force / params.sprung_mass
    wheel_accel = (tire_force - suspension_force) / params.unsprung_mass

    next_state = QuarterCarState(
        time=state.time + dt,
        v_body=state.v_body + body_accel * dt,
        v_wheel=state.v_wheel + wheel_accel * dt,
    )
    next_state.z_body = state.z_body + next_state.v_body * dt
    next_state.z_wheel = state.z_wheel + next_state.v_wheel * dt

    travel = abs(next_state.z_wheel - next_state.z_body)
    relative_velocity = abs(next_state.v_wheel - next_state.v_body)
    power = params.damping_coeff * relative_velocity**2

    return next_state, {
        "time": next_state.time,
        "body_accel": body_accel,
        "wheel_accel": wheel_accel,
        "road_input_cm": road_input * 100,
        "body_disp_cm": next_state.z_body * 100,
        "suspension_travel_cm": travel * 100,
        "dissipated_energy": power * dt,
        "distance_m": (params.velocity / 3.6) * dt,
    }


def run_live_step(params: VehicleParams, road_profile: str, road_amplitude: float) -> None:
    next_state, sample = simulate_step(
        st.session_state.model_state,
        params,
        road_profile,
        road_amplitude,
    )
    st.session_state.model_state = next_state
    st.session_state.accel_history.append(abs(sample["body_accel"]))
    st.session_state.travel_history.append(sample["suspension_travel_cm"])
    st.session_state.chart_data.append(sample)
    st.session_state.dissipated_energy += sample["dissipated_energy"]
    st.session_state.total_distance += sample["distance_m"]


def calculate_comfort_index(accelerations: Iterable[float]) -> float:
    accel_array = np.array(list(accelerations), dtype=float)
    if accel_array.size == 0:
        return 100.0

    rms = float(np.sqrt(np.mean(accel_array**2)))
    if rms < 0.315:
        index = 100 - (rms / 0.315) * 10
    elif rms < 0.63:
        index = 90 - ((rms - 0.315) / 0.315) * 30
    elif rms < 1.0:
        index = 60 - ((rms - 0.63) / 0.37) * 30
    elif rms < 2.0:
        index = 30 - ((rms - 1.0) / 1.0) * 20
    else:
        index = 10 - ((rms - 2.0) / 2.0) * 10

    return float(np.clip(index, 0.0, 100.0))


def get_comfort_rating(index: float) -> tuple[str, str]:
    if index >= 90:
        return "Excellent", "#10b981"
    if index >= 70:
        return "Good", "#14b8a6"
    if index >= 50:
        return "Fair", "#f59e0b"
    if index >= 30:
        return "Poor", "#f97316"
    return "Very Poor", "#ef4444"


def calculate_frequency_response(accel_history: Iterable[float], dt: float = DT_SECONDS) -> tuple[np.ndarray, np.ndarray]:
    accel_array = np.array(list(accel_history), dtype=float)
    if accel_array.size < 50:
        return np.array([]), np.array([])

    accel_array = accel_array - np.mean(accel_array)
    fft_vals = np.fft.rfft(accel_array)
    fft_freq = np.fft.rfftfreq(accel_array.size, dt)
    magnitude = np.abs(fft_vals) / accel_array.size
    mask = (fft_freq > 0) & (fft_freq <= 20)
    return fft_freq[mask], magnitude[mask]


def grid_search_optimization(
    params: VehicleParams,
    spring_values: np.ndarray,
    damping_values: np.ndarray,
    road_profile: str,
    road_amplitude: float,
    steps: int = 350,
) -> tuple[dict[str, float], float, pd.DataFrame]:
    best_score = -np.inf
    best_params: dict[str, float] = {}
    rows = []

    for spring in spring_values:
        for damping in damping_values:
            test_params = VehicleParams(
                sprung_mass=params.sprung_mass,
                unsprung_mass=params.unsprung_mass,
                spring_stiffness=float(spring),
                damping_coeff=float(damping),
                tire_stiffness=params.tire_stiffness,
                tire_damping=params.tire_damping,
                velocity=params.velocity,
            )
            state = QuarterCarState()
            accel_local = []

            for _ in range(steps):
                state, sample = simulate_step(
                    state,
                    test_params,
                    road_profile,
                    road_amplitude,
                )
                accel_local.append(abs(sample["body_accel"]))

            comfort = calculate_comfort_index(accel_local)
            rows.append(
                {
                    "spring_stiffness": float(spring),
                    "damping_coeff": float(damping),
                    "comfort_index": comfort,
                }
            )

            if comfort > best_score:
                best_score = comfort
                best_params = {
                    "spring_stiffness": float(spring),
                    "damping_coeff": float(damping),
                }

    return best_params, float(best_score), pd.DataFrame(rows)


def render_header() -> None:
    st.markdown(
        """
        <h1 style="text-align:center; color:#f8fafc; margin-bottom:0.2rem;">
            Suspension Comfort Analyzer
        </h1>
        <div class="app-subtitle">
            Real-time quarter-car simulation with ISO-style comfort scoring and tuning support
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> tuple[VehicleParams, str, float, dict[str, bool]]:
    st.sidebar.header("Suspension Parameters")
    left, right = st.sidebar.columns(2)

    with left:
        sprung_mass = st.slider("Sprung mass (kg)", 250, 1600, 900, 25)
        spring_stiffness = st.slider("Spring stiffness (kN/m)", 10, 180, 42, 2) * 1000
    with right:
        damping_coeff = st.slider("Damping (Ns/m)", 500, 15000, 3800, 100)
        velocity = st.slider("Velocity (km/h)", 20, 200, 80, 5)

    st.sidebar.header("Road Conditions")
    road_profile = st.sidebar.selectbox(
        "Road profile",
        ["Smooth Road", "Wavy Road", "Rough Road", "Speed Bumps", "Pothole", "Random Road"],
    )
    road_amplitude = st.sidebar.slider("Road severity", 0.5, 2.0, 1.0, 0.1)

    descriptions = {
        "Smooth Road": "Low-amplitude highway-like input.",
        "Wavy Road": "Longer rolling undulations plus a small harmonic ripple.",
        "Rough Road": "Higher-frequency uneven surface input.",
        "Speed Bumps": "Repeated raised obstacles scaled by vehicle speed.",
        "Pothole": "Repeated downward impacts scaled by vehicle speed.",
        "Random Road": "Noise-like surface with a light periodic component.",
    }
    st.sidebar.info(descriptions[road_profile])

    with st.sidebar.expander("Advanced settings", expanded=False):
        unsprung_mass = st.slider("Unsprung mass (kg)", 35, 250, 95, 5)
        tire_stiffness = st.slider("Tire stiffness (kN/m)", 100, 1000, 260, 10) * 1000
        tire_damping = st.slider("Tire damping (Ns/m)", 100, 3500, 900, 50)
        show_frequency = st.checkbox("Show frequency analysis", value=True)
        show_travel = st.checkbox("Show suspension travel", value=True)
        show_optimizer = st.checkbox("Enable optimization tool", value=True)

    params = VehicleParams(
        sprung_mass=float(sprung_mass),
        unsprung_mass=float(unsprung_mass),
        spring_stiffness=float(spring_stiffness),
        damping_coeff=float(damping_coeff),
        tire_stiffness=float(tire_stiffness),
        tire_damping=float(tire_damping),
        velocity=float(velocity),
    )
    flags = {
        "show_frequency": show_frequency,
        "show_travel": show_travel,
        "show_optimizer": show_optimizer,
    }
    return params, road_profile, road_amplitude, flags


def render_controls(params: VehicleParams, road_profile: str, road_amplitude: float, show_optimizer: bool) -> None:
    st.divider()
    col_start, col_reset, col_export, col_optimize = st.columns([1.0, 1.0, 1.4, 2.0])

    with col_start:
        label = "Pause" if st.session_state.is_running else "Start"
        if st.button(label, use_container_width=True, type="primary"):
            st.session_state.is_running = not st.session_state.is_running
            st.rerun()

    with col_reset:
        if st.button("Reset", use_container_width=True):
            reset_simulation()
            st.rerun()

    with col_export:
        df = pd.DataFrame(list(st.session_state.chart_data))
        st.download_button(
            "Export CSV",
            data=df.to_csv(index=False) if not df.empty else "",
            file_name="suspension_data.csv",
            mime="text/csv",
            disabled=df.empty,
            use_container_width=True,
        )

    if show_optimizer:
        with col_optimize:
            if st.button("Optimize for comfort", use_container_width=True):
                with st.spinner("Running tuning sweep..."):
                    spring_values = np.arange(12_000, 90_001, 4_000)
                    damping_values = np.arange(800, 9_001, 400)
                    best_params, best_score, result_df = grid_search_optimization(
                        params,
                        spring_values,
                        damping_values,
                        road_profile,
                        road_amplitude,
                    )
                st.session_state.optimization_runs.append(
                    {
                        "road_profile": road_profile,
                        "velocity": params.velocity,
                        "best_score": best_score,
                        "best_spring": best_params["spring_stiffness"],
                        "best_damping": best_params["damping_coeff"],
                        "result_df": result_df,
                    }
                )
                st.success(
                    "Best comfort %.1f/100 at %.0f kN/m spring and %.0f Ns/m damping."
                    % (
                        best_score,
                        best_params["spring_stiffness"] / 1000,
                        best_params["damping_coeff"],
                    )
                )


def render_metrics(params: VehicleParams) -> tuple[float, float, float, float]:
    accel_array = np.array(list(st.session_state.accel_history), dtype=float)
    travel_array = np.array(list(st.session_state.travel_history), dtype=float)

    if accel_array.size:
        rms_accel = float(np.sqrt(np.mean(accel_array**2)))
        peak_accel = float(np.max(accel_array))
        vdv = float(np.mean(np.abs(accel_array) ** 4) ** 0.25)
        comfort_index = calculate_comfort_index(accel_array)
    else:
        rms_accel = peak_accel = vdv = 0.0
        comfort_index = 100.0

    avg_travel = float(np.mean(travel_array)) if travel_array.size else 0.0
    max_travel = float(np.max(travel_array)) if travel_array.size else 0.0
    comfort_text, comfort_color = get_comfort_rating(comfort_index)

    st.divider()
    cols = st.columns(5)
    with cols[0]:
        st.metric("Comfort Index", f"{comfort_index:.1f}/100")
        st.markdown(
            f'<span class="comfort-pill" style="background:{comfort_color};">{comfort_text}</span>',
            unsafe_allow_html=True,
        )
    cols[1].metric("RMS Accel", f"{rms_accel:.3f} m/s^2", delta="Low" if rms_accel < 0.5 else "High")
    cols[2].metric("Peak Accel", f"{peak_accel:.3f} m/s^2")
    cols[3].metric("Suspension Travel", f"{avg_travel:.1f} cm", delta=f"Max {max_travel:.1f} cm")
    cols[4].metric(
        "Dissipated Energy",
        f"{st.session_state.dissipated_energy / 1000:.1f} kJ",
        delta=f"{st.session_state.total_distance:.0f} m",
    )

    damping_ratio = params.damping_coeff / (2 * np.sqrt(params.spring_stiffness * params.sprung_mass))
    if damping_ratio < 0.25:
        st.warning("Low damping ratio: the body response may oscillate after impacts.")
    elif damping_ratio > 0.75:
        st.info("High damping ratio: ride harshness may increase on rough surfaces.")
    if max_travel > 8:
        st.error(f"High suspension travel ({max_travel:.1f} cm): bottoming risk is elevated.")
    if rms_accel > 1.0:
        st.error("High vibration level detected. Comfort is likely poor.")

    return comfort_index, rms_accel, vdv, max_travel


def render_response_chart(show_travel: bool) -> None:
    st.divider()
    st.subheader("Dynamic Response")

    if not st.session_state.chart_data:
        st.info("Press Start to begin the simulation.")
        return

    df = pd.DataFrame(list(st.session_state.chart_data))
    rows = 2 if show_travel else 1
    fig = make_subplots(
        rows=rows,
        cols=1,
        subplot_titles=("Acceleration Response", "Suspension Travel") if show_travel else ("Acceleration Response",),
        vertical_spacing=0.14,
        row_heights=[0.62, 0.38] if show_travel else [1.0],
    )
    fig.add_trace(
        go.Scatter(
            x=df["time"],
            y=df["body_accel"],
            mode="lines",
            name="Body acceleration",
            line=dict(color="#14b8a6", width=2),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df["time"],
            y=df["wheel_accel"],
            mode="lines",
            name="Wheel acceleration",
            line=dict(color="#f59e0b", width=1.5),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df["time"],
            y=df["road_input_cm"],
            mode="lines",
            name="Road input",
            line=dict(color="#94a3b8", width=1, dash="dash"),
        ),
        row=1,
        col=1,
    )

    if show_travel:
        fig.add_trace(
            go.Scatter(
                x=df["time"],
                y=df["suspension_travel_cm"],
                mode="lines",
                name="Suspension travel",
                line=dict(color="#38bdf8", width=2),
                fill="tozeroy",
                fillcolor="rgba(56, 189, 248, 0.18)",
            ),
            row=2,
            col=1,
        )
        fig.update_yaxes(title_text="Travel (cm)", row=2, col=1)

    fig.update_xaxes(title_text="Time (s)", row=rows, col=1)
    fig.update_yaxes(title_text="Acceleration (m/s^2)", row=1, col=1)
    fig.update_layout(
        height=620 if show_travel else 420,
        template="plotly_dark",
        hovermode="x unified",
        showlegend=True,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.55)",
        legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=70, b=30),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_frequency_analysis(params: VehicleParams, show_frequency: bool) -> None:
    if not show_frequency or len(st.session_state.accel_history) < 50:
        return

    freqs, magnitude = calculate_frequency_response(st.session_state.accel_history)
    if freqs.size == 0:
        return

    left, right = st.columns([2, 1])
    with left:
        st.subheader("Frequency Spectrum")
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=freqs,
                y=magnitude,
                mode="lines",
                line=dict(color="#f97316", width=2),
                fill="tozeroy",
                fillcolor="rgba(249, 115, 22, 0.22)",
                name="Magnitude",
            )
        )
        fig.add_vrect(
            x0=4,
            x1=8,
            fillcolor="rgba(245, 158, 11, 0.13)",
            layer="below",
            line_width=0,
            annotation_text="4-8 Hz sensitive band",
            annotation_position="top left",
        )
        fig.update_layout(
            height=320,
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,42,0.55)",
            xaxis_title="Frequency (Hz)",
            yaxis_title="Magnitude",
            margin=dict(l=20, r=20, t=40, b=30),
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Frequency Stats")
        dominant_frequency = float(freqs[int(np.argmax(magnitude))])
        natural_frequency = float(np.sqrt(params.spring_stiffness / params.sprung_mass) / (2 * np.pi))
        damping_ratio = float(params.damping_coeff / (2 * np.sqrt(params.spring_stiffness * params.sprung_mass)))
        st.metric("Dominant frequency", f"{dominant_frequency:.2f} Hz")
        st.metric("Natural frequency", f"{natural_frequency:.2f} Hz")
        st.metric(
            "Damping ratio",
            f"{damping_ratio:.3f}",
            delta="Balanced" if 0.25 <= damping_ratio <= 0.75 else "Review",
        )


def render_optimizer_history() -> None:
    if not st.session_state.optimization_runs:
        return

    latest = st.session_state.optimization_runs[-1]
    result_df = latest["result_df"]
    heatmap_df = result_df.pivot(
        index="spring_stiffness",
        columns="damping_coeff",
        values="comfort_index",
    )

    st.divider()
    st.subheader("Latest Optimization Sweep")
    fig = go.Figure(
        data=go.Heatmap(
            x=heatmap_df.columns,
            y=heatmap_df.index / 1000,
            z=heatmap_df.values,
            colorscale="Viridis",
            colorbar=dict(title="Comfort"),
        )
    )
    fig.update_layout(
        height=360,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.55)",
        xaxis_title="Damping (Ns/m)",
        yaxis_title="Spring stiffness (kN/m)",
        margin=dict(l=20, r=20, t=35, b=35),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_model_notes(params: VehicleParams, comfort_index: float, rms_accel: float, vdv: float) -> None:
    st.divider()
    left, right = st.columns(2)

    with left:
        st.subheader("Quarter-Car Model")
        st.code(
            f"""Body:
m1 * z1_ddot = k1(z2 - z1) + c1(z2_dot - z1_dot)

Wheel:
m2 * z2_ddot = k2(zr - z2) + c2(zr_dot - z2_dot)
               - k1(z2 - z1) - c1(z2_dot - z1_dot)

Current parameters:
m1 = {params.sprung_mass:.0f} kg
m2 = {params.unsprung_mass:.0f} kg
k1 = {params.spring_stiffness / 1000:.0f} kN/m
c1 = {params.damping_coeff:.0f} Ns/m
k2 = {params.tire_stiffness / 1000:.0f} kN/m
c2 = {params.tire_damping:.0f} Ns/m
""",
            language="text",
        )

    with right:
        comfort_text, comfort_color = get_comfort_rating(comfort_index)
        st.subheader("Comfort Reference")
        st.markdown(
            f"""
            <b>Current comfort:</b>
            <span style="color:{comfort_color}; font-weight:700;">{comfort_text}</span><br>
            <b>RMS acceleration:</b> {rms_accel:.3f} m/s^2<br>
            <b>VDV-style value:</b> {vdv:.3f} m/s^1.75

            - Less than 0.315 m/s^2: not uncomfortable
            - 0.315 to 0.63 m/s^2: slightly uncomfortable
            - 0.63 to 1.0 m/s^2: fairly uncomfortable
            - 1.0 to 2.0 m/s^2: uncomfortable
            - More than 2.0 m/s^2: very uncomfortable
            """,
            unsafe_allow_html=True,
        )


def render_footer() -> None:
    st.markdown(
        f"""
        <div style="text-align:center; color:#94a3b8; padding:18px; font-size:0.9rem;">
            Model: 2-DOF quarter-car | Integration: semi-implicit Euler, dt={DT_SECONDS * 1000:.0f} ms |
            Time: {st.session_state.model_state.time:.2f}s |
            Samples: {len(st.session_state.chart_data)} |
            Status: {"Running" if st.session_state.is_running else "Paused"} |
            Distance: {st.session_state.total_distance:.1f} m
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    configure_page()
    initialize_state()
    render_header()

    params, road_profile, road_amplitude, flags = render_sidebar()
    render_controls(params, road_profile, road_amplitude, flags["show_optimizer"])

    if st.session_state.is_running:
        for _ in range(STEPS_PER_REFRESH):
            run_live_step(params, road_profile, road_amplitude)
        time.sleep(0.03)
        st.rerun()

    comfort_index, rms_accel, vdv, _ = render_metrics(params)
    render_response_chart(flags["show_travel"])
    render_frequency_analysis(params, flags["show_frequency"])
    render_optimizer_history()
    render_model_notes(params, comfort_index, rms_accel, vdv)
    render_footer()


if __name__ == "__main__":
    main()
