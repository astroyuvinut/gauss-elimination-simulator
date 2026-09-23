"""Interactive (rotatable) Plotly view of the final system, shown after the last clip."""
from __future__ import annotations

import plotly.graph_objects as go

from geometry import clip_line, plane_polygon, view_range
from pipeline import result_info
from scenes import BG, ROW_COLORS
from solver import Result, fmt


def _equation(row) -> str:
    """'x - 3y + z = 4' style text, skipping zero terms and unit coefficients."""
    out = ""
    for c, v in zip(row[:3], "xyz"):
        if c == 0:
            continue
        mag = "" if abs(c) == 1 else fmt(abs(c))
        out += (("-" if c < 0 else "") if not out else (" - " if c < 0 else " + ")) + mag + v
    return f"{out} = {fmt(row[3])}"


def final_figure(res: Result) -> go.Figure:
    R = view_range([res.initial] + [f.after for f in res.frames], res.solution)
    fig = go.Figure()
    for i, row in enumerate(res.final):
        poly = plane_polygon(row, R)
        if poly is None:
            continue
        xs, ys, zs = zip(*poly)
        k = len(poly)  # fan triangulation of the convex clipped polygon
        colour = ROW_COLORS[i % 3].to_hex()
        eq = _equation(row)
        fig.add_trace(go.Mesh3d(x=xs, y=ys, z=zs, i=[0] * (k - 2), j=list(range(1, k - 1)),
                                k=list(range(2, k)), color=colour, opacity=0.45, flatshading=True,
                                name=f"R{i + 1}: {eq}", showlegend=True,
                                hovertemplate="x=%{x:.2f}<br>y=%{y:.2f}<br>z=%{z:.2f}"
                                              f"<extra>R{i + 1}</extra>"))
    info = result_info(res)
    if info["kind"] == "unique":
        x, y, z = info["point"]
        fig.add_trace(go.Scatter3d(x=[x], y=[y], z=[z], mode="markers", name=info["caption"],
                                   marker=dict(size=7, color="#e8d44d", line=dict(color="white", width=2))))
    elif info["kind"] == "line":
        seg = clip_line(info["point"], info["direction"], R)
        if seg is not None:
            (x0, y0, z0), (x1, y1, z1) = seg
            fig.add_trace(go.Scatter3d(x=[x0, x1], y=[y0, y1], z=[z0, z1], mode="lines",
                                       name=info["caption"], line=dict(color="#e8d44d", width=8)))
    axis = dict(range=[-R, R], backgroundcolor=BG, gridcolor="#2a2e3a", zerolinecolor="#6b7080",
                color="#c9ccd6")
    fig.update_layout(
        scene=dict(xaxis=dict(axis, title="x"), yaxis=dict(axis, title="y"), zaxis=dict(axis, title="z"),
                   aspectmode="cube", camera=dict(eye=dict(x=1.5, y=-1.6, z=1.0))),
        paper_bgcolor=BG, font=dict(color="#c9ccd6"), margin=dict(l=0, r=0, t=30, b=0), height=520,
        title=dict(text=info["caption"] + "  ·  drag to rotate, scroll to zoom", x=0.02, font=dict(size=14)),
        legend=dict(x=0.01, y=0.92, bgcolor="rgba(0,0,0,0)"),
    )
    return fig
