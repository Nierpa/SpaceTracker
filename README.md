# 🛰️ SpaceTracker

An interactive data science project to simulate satellite orbits, detect potential collisions, and predict collision risk using machine learning.

---

## 🌍 Overview

SpaceTracker combines:

- Orbital mechanics simulation (TLE data)
- Collision detection (conjunction analysis)
- Machine learning (risk prediction)
- Interactive 3D visualization (Streamlit)

It also simulates space debris events and explores the **Kessler syndrome**.

---

## ✨ Features

- 🛰️ Real satellite data (TLE / NORAD)
- 📡 Orbit propagation in 3D
- 💥 Collision detection between objects
- 🧠 Machine learning risk prediction
- ☄️ Debris event simulation
- 🌌 Interactive 3D visualization
- 🔄 Live satellite data fetching

---

## 📊 Data Science Pipeline

1. Load TLE satellite data
2. Propagate orbits over time
3. Detect close approaches (conjunctions)
4. Engineer features:
   - Distance
   - Relative velocity
5. Generate synthetic collision labels
6. Train a classification model
7. Predict collision risk

---

## 📓 Notebook

Detailed explanation available in:
notebooks/satellite_collision_analysis.ipynb

This notebook explains:

- Data preparation
- Feature engineering
- Model training
- Limitations

---

## 🖥️ Streamlit App

Run the interactive dashboard:

streamlit run dashboard/app.py

---

## 🖳 Installation
git clone https://github.com/your-username/SpaceTracker.git
cd SpaceTracker

pip install -r requirements.txt

---

## 📁 Data

TLE files are stored in:
data/raw/
You can also fetch live data directly from the app.

>>>>>>> d4ce27f (Initial commit - SpaceTracker: satellite collision prediction with ML and Streamlit dashboard)
