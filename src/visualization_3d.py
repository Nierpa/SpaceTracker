"""
3D visualization with animation.
"""

import plotly.graph_objects as go
import numpy as np

def plot_animated_orbits(positions):

    fig = go.Figure()

    for sat in positions:

        pos = positions[sat]

        fig.add_trace(
            go.Scatter3d(
                x=pos[0],
                y=pos[1],
                z=pos[2],
                mode='lines',
                name=sat
            )
        )

    # Add simple animation feel (frames)
    frames = []

    for t in range(0, pos.shape[1], 5):

        frame_data = []

        for sat in positions:

            p = positions[sat]

            frame_data.append(
                go.Scatter3d(
                    x=[p[0][t]],
                    y=[p[1][t]],
                    z=[p[2][t]],
                    mode='markers'
                )
            )

        frames.append(go.Frame(data=frame_data))

    fig.frames = frames

    return fig
    
    
def plot_3d_satellites(positions, risk_dict=None):
    """
    Plot satellites in 3D with optional risk coloring.
    """

    fig = go.Figure()

    for sat in positions:

        pos = positions[sat]

        # last position (instant view)
        x, y, z = pos[0][-1], pos[1][-1], pos[2][-1]

        color = "blue"

        if risk_dict and sat in risk_dict:
            if risk_dict[sat] > 0.7:
                color = "red"

        fig.add_trace(
            go.Scatter3d(
                x=[x], y=[y], z=[z],
                mode='markers',
                marker=dict(size=4, color=color),
                name=sat
            )
        )

    return fig
