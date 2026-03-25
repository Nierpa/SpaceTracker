import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
DATA_DIR = PROJECT_ROOT / "data" / "raw"

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px

from src.tle_loader import load_multiple_tle
from src.orbit_propagation import propagate_orbits
from src.conjunctions import compute_conjunctions
from src.features import compute_relative_velocity
from src.ml_model import train_model, predict_risk
from src.debris_simulation import generate_debris
from src.get_data import fetch_tle_data


# CONFIG
st.set_page_config(page_title="Satellite Collision Intelligence", layout="wide")
st.title("🛰️ Satellite Collision Intelligence System")

# SIDEBAR
st.sidebar.markdown("## 🌐 Live Satellite Data")

tle_sources = {
    "Active Satellites": "https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle",
    "Starlink": "https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=tle",
    "Space Stations": "https://celestrak.org/NORAD/elements/gp.php?GROUP=stations&FORMAT=tle",
    "Debris": "https://celestrak.org/NORAD/elements/gp.php?GROUP=debris&FORMAT=tle"
}

selected_source = st.sidebar.selectbox("Select Live Source", list(tle_sources.keys()))

auto_refresh = st.sidebar.checkbox("🔄 Auto-refresh (60s)")
fetch_button = st.sidebar.button("📡 Fetch Live Sats")

st.sidebar.header("Simulation Settings")

n_sat = st.sidebar.slider("Satellites per dataset", 10, 200, 50)
duration = st.sidebar.slider("Simulation duration (minutes)", 30, 300, 120)

# Cache fetching to avoid hitting API limits
@st.cache_data(ttl=60)
def cached_fetch(url, output):
    return fetch_tle_data(url, output)

live_file = str(DATA_DIR / f"live_{selected_source.replace(' ', '_')}.tle")

if fetch_button:
    success = fetch_tle_data(tle_sources[selected_source], live_file)
    if success:
        st.sidebar.success("Live data updated!")

# Auto-refresh
if auto_refresh:
    cached_fetch(tle_sources[selected_source], live_file)

datasets = st.sidebar.multiselect(
    "Select datasets",
    [
        "science_Satellites.tle",
        "military_Satellites.tle",
        "debris_Standards.tle",
        "debris_30.tle",
        "debris_Lancement.tle",
        "satellites_Standards.tle",
        f"live_{selected_source.replace(' ', '_')}.tle"
    ],
    default=["science_Satellites.tle"]
)

simulate_event = st.sidebar.checkbox("💥 Trigger Space Event")
n_debris = st.sidebar.slider("Debris count", 5, 100, 20)


# DATA
files = [str(DATA_DIR / d) for d in datasets]
valid_files = []

for f in files:
    if Path(f).exists():
        valid_files.append(f)
    else:
        st.warning(f"File not found: {f}")

files = valid_files
satellites = load_multiple_tle(files, max_per_file=n_sat)
positions = propagate_orbits(satellites, minutes=duration)

# Object selection
objects = list(positions.keys())

if "selected_object" not in st.session_state:
    st.session_state.selected_object = objects[0]

if st.session_state.selected_object not in objects:
    st.session_state.selected_object = objects[0]

selected_object = st.sidebar.selectbox(
    "Select object",
    objects,
    index=objects.index(st.session_state.selected_object),
    key="selected_object_box"  # ✅ IMPORTANT
)

# save selection 
st.session_state.selected_object = selected_object

# debris
if simulate_event:
    debris_positions = generate_debris(n_debris)
    positions.update(debris_positions)

# collisions
events, fragments = compute_conjunctions(
    positions,
    threshold_km=50,
    generate_fragments=simulate_event
)

positions.update(fragments)

# ML DATASET
features = []
for event in events:

    posA = positions[event["sat1"]]
    posB = positions[event["sat2"]]

    velocity = compute_relative_velocity(posA, posB)

    features.append({
        "sat1": event["sat1"],
        "sat2": event["sat2"],
        "distance": event["min_distance"],
        "relative_velocity": velocity
    })

df_ml = pd.DataFrame(features)

if len(df_ml) == 0:
    st.warning("No conjunctions detected.")
    st.stop()

# NOISE
df_ml["distance"] += np.random.normal(0, 5, len(df_ml))
df_ml["relative_velocity"] += np.random.normal(0, 0.5, len(df_ml))

# LABELS
threshold_dist = df_ml["distance"].quantile(0.5)
threshold_vel = df_ml["relative_velocity"].median()

df_ml["collision"] = (
    (df_ml["distance"] < threshold_dist) &
    (df_ml["relative_velocity"] > threshold_vel)
).astype(int)

# BALANCING
counts = df_ml["collision"].value_counts()

if df_ml["collision"].nunique() < 2 or counts.min() < 2:

    st.warning("⚙️ Adjusting dataset to ensure ML training...")

    df_ml = df_ml.sort_values(["distance", "relative_velocity"]).reset_index(drop=True)

    n = len(df_ml)

    df_ml["collision"] = 0

    split = max(2, n // 2)

    df_ml.loc[:split-1, "collision"] = 1
    df_ml.loc[split:split+1, "collision"] = 0

    counts = df_ml["collision"].value_counts()
    st.write("Balanced dataset:", counts)

# MODEL OR FALLBACK
use_ml = False

if len(df_ml) < 10 or counts.min() < 2:

    st.warning("""
    ⚠️ Not enough data to train a reliable ML model.

    Try:
    - increasing satellites
    - increasing simulation duration
    - enabling debris / space event
    """)

    # fallback risk 
    df_ml["risk"] = (
        1 / (df_ml["distance"] + 1)
        * df_ml["relative_velocity"]
    )

else:
    model, _ = train_model(df_ml)
    df_ml["risk"] = predict_risk(model, df_ml)
    use_ml = True

# RISK MAP
risk_dict = {}
for _, row in df_ml.iterrows():
    risk_dict[row["sat1"]] = row["risk"]
    risk_dict[row["sat2"]] = row["risk"]


# 3D VIEW 
tab1, tab2, tab3 = st.tabs(["🌍 3D View", "📊 Risk Analysis", "📋 Data"])

with tab1:

    fig = go.Figure()

    # Earth 
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode="markers",
        marker=dict(size=10, color="blue"),
        name="Earth"
    ))

    # Milky Way background
    img = Image.open(PROJECT_ROOT/"src/assets/milkyWay.jpeg")

    fig.update_layout(
        images=[
            dict(
                source=img,
                xref="paper",
                yref="paper",
                x=0,
                y=1,
                sizex=1,
                sizey=1,
                sizing="stretch",
                opacity=0.4,
                layer="below"
            )
        ]
    )

    for sat in positions:

        pos = positions[sat]

        # STYLE 
        if "DEBRIS" in sat:
            color = "gray"
            size = 2
            opacity = 0.3

        elif "FRAG" in sat:
            color = "yellow"
            size = 6
            opacity = 1

        elif "ISS" in sat.upper():
            color = "red"
            size = 9
            opacity = 1

        elif risk_dict.get(sat, 0) > 0.7:
            color = "orange"
            size = 5
            opacity = 0.9

        else:
            color = "green"
            size = 4
            opacity = 0.7

        # highlight sélection
        if sat == selected_object:
            size = 9
            color = "white"

        # Trajectories
        tail = 50

        fig.add_trace(
            go.Scatter3d(
                x=pos[0][-tail:],
                y=pos[1][-tail:],
                z=pos[2][-tail:],
                mode='lines',
                opacity=opacity,
                line=dict(width=1, color=color),
                name=sat
            )
        )
    fig.update_layout(
    height=900,
    margin=dict(l=0, r=0, t=0, b=0),
    scene=dict(
        camera=dict(
            eye=dict(x=1.5, y=1.5, z=1.2)
        )
    )
)
    st.plotly_chart(fig, use_container_width=True)

# LIVE COLLISION ANALYSIS
if selected_object and selected_object in positions:

    st.subheader(f"☄️ Risk Analysis for: {selected_object}")

    risks = []

    for other in positions:

        if other == selected_object:
            continue

        try:
            posA = positions[selected_object]
            posB = positions[other]

            dist = np.min(np.linalg.norm(posA - posB, axis=0))
            vel = compute_relative_velocity(posA, posB)

            if use_ml:
                risk_score = predict_risk(model, pd.DataFrame([{
                    "distance": dist,
                    "relative_velocity": vel
                }]))[0]
            else:
                risk_score = (1 / (dist + 1)) * vel

            risks.append({
                "object": other,
                "distance": dist,
                "velocity": vel,
                "risk": risk_score
            })

        except:
            continue

    df_risk_live = pd.DataFrame(risks)

    if not df_risk_live.empty and "risk" in df_risk_live.columns:
        df_risk_live = df_risk_live.sort_values("risk", ascending=False)
        st.dataframe(df_risk_live.head(10), use_container_width=True)
    else:
        st.warning("No risk data available.")

# ANALYSIS
with tab2:

    fig_hist = px.histogram(df_ml, x="risk", nbins=30)
    st.plotly_chart(fig_hist, use_container_width=True)

    fig_scatter = px.scatter(
        df_ml,
        x="distance",
        y="relative_velocity",
        color="risk",
        color_continuous_scale="RdYlGn_r"
    )

    st.plotly_chart(fig_scatter, use_container_width=True)

# DATA
with tab3:

    st.dataframe(
        df_ml.sort_values("risk", ascending=False).head(20),
        use_container_width=True
    )
