"""Map and chart helpers for the Streamlit app."""
from __future__ import annotations

from datetime import date, timedelta

import ee
import folium
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from pamrofi.agro import ADVISORY_COLORS, ADVISORY_LABELS, STAGE_NAMES, STRESS_COLORS
from pamrofi.viz import STAGE_COLORS


# ------------------------------------------------------------------ maps
def base_map(center: tuple[float, float], zoom: int) -> folium.Map:
    m = folium.Map(location=center, zoom_start=zoom, control_scale=True, tiles=None)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery", name="Satellite", overlay=False,
    ).add_to(m)
    folium.TileLayer("OpenStreetMap", name="Streets", overlay=False).add_to(m)
    return m


def add_ee_layer(m: folium.Map, tile_url: str, name: str, shown: bool = True,
                 opacity: float = 0.85) -> None:
    folium.TileLayer(tiles=tile_url, attr="Google Earth Engine", name=name, overlay=True,
                     control=True, show=shown, opacity=opacity).add_to(m)


def add_roi_outline(m: folium.Map, roi_geojson: dict) -> None:
    folium.GeoJson(roi_geojson, name="Region of interest",
                   style_function=lambda _: {"color": "#ec4899", "weight": 2, "fillOpacity": 0}
                   ).add_to(m)


def add_marker(m: folium.Map, lat: float, lon: float) -> None:
    folium.CircleMarker([lat, lon], radius=7, color="#111827", fill=True,
                        fill_color="#facc15", fill_opacity=1, tooltip="Selected field").add_to(m)


def add_legend(m: folium.Map, title: str, items: dict[str, str]) -> None:
    rows = "".join(
        f'<div style="display:flex;align-items:center;gap:6px;margin:2px 0">'
        f'<span style="width:14px;height:14px;border-radius:3px;background:{c};'
        f'display:inline-block;border:1px solid #0003"></span>{label}</div>'
        for label, c in items.items()
    )
    html = (f'<div style="position:absolute;bottom:24px;left:12px;z-index:9999;'
            f'background:rgba(255,255,255,.92);padding:8px 10px;border-radius:8px;'
            f'font:12px/1.3 system-ui,sans-serif;color:#111;box-shadow:0 1px 4px #0003">'
            f'<b>{title}</b>{rows}</div>')
    m.get_root().html.add_child(folium.Element(html))


def finish(m: folium.Map) -> folium.Map:
    folium.LayerControl(collapsed=True).add_to(m)
    return m


# ---------------------------------------------------------------- dates
def period_starts(start: str, end: str, step: int) -> list[date]:
    s, e = date.fromisoformat(start), date.fromisoformat(end)
    out, d = [], s
    while d + timedelta(days=step) <= e + timedelta(days=1):
        out.append(d)
        d += timedelta(days=step)
    return out


# --------------------------------------------------------------- charts
def _stage_shapes(df: pd.DataFrame) -> list[dict]:
    shapes = []
    if df.empty or "PAM_stage" not in df:
        return shapes
    prev, start = None, None
    for d, s in zip(df["date"], df["PAM_stage"].fillna(0).astype(int)):
        if s != prev:
            if prev is not None:
                shapes.append((start, d, prev))
            prev, start = s, d
    shapes.append((start, df["date"].iloc[-1], prev))
    return [dict(type="rect", xref="x", yref="paper", x0=a, x1=b, y0=0, y1=1,
                 fillcolor=STAGE_COLORS[s], opacity=0.18, line_width=0, layer="below")
            for a, b, s in shapes if s]


def trajectory_chart(df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=df["date"], y=df["NDVI"], name="NDVI", mode="lines+markers",
                             line=dict(color="#15803d", width=2.5)), secondary_y=False)
    cloudy = df[df["obs"] == 0]
    fig.add_trace(go.Scatter(x=cloudy["date"], y=cloudy["NDVI"], name="NDVI (gap-filled)",
                             mode="markers", marker=dict(color="#9ca3af", size=8, symbol="x")),
                  secondary_y=False)
    fig.add_trace(go.Scatter(x=df["date"], y=df["VH"], name="VH (dB)", mode="lines",
                             line=dict(color="#2563eb", dash="dot")), secondary_y=True)
    fig.update_layout(title="Crop trajectory with PAM growth stages (shaded)",
                      shapes=_stage_shapes(df), height=330, margin=dict(l=10, r=10, t=40, b=10),
                      legend=dict(orientation="h", y=-0.2))
    fig.update_yaxes(title_text="NDVI", range=[0, 1], secondary_y=False)
    fig.update_yaxes(title_text="VH backscatter (dB)", secondary_y=True)
    return fig


def stress_chart(df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    colors = [STRESS_COLORS.get(int(s), "#d1d5db") if pd.notna(s) else "#e5e7eb"
              for s in df["stress"]]
    fig.add_trace(go.Bar(x=df["date"], y=[1] * len(df), marker_color=colors, opacity=0.35,
                         name="Stress level", hovertext=df["stress"]), secondary_y=False)
    fig.add_trace(go.Scatter(x=df["date"], y=df["VCI"], name="VCI (0–100)",
                             line=dict(color="#15803d")), secondary_y=True)
    fig.add_trace(go.Scatter(x=df["date"], y=df["SMI"] * 100, name="SMI × 100",
                             line=dict(color="#92400e")), secondary_y=True)
    fig.update_layout(title="Moisture stress history (bars coloured by stress level)",
                      height=300, margin=dict(l=10, r=10, t=40, b=10), bargap=0,
                      legend=dict(orientation="h", y=-0.25))
    fig.update_yaxes(visible=False, secondary_y=False)
    fig.update_yaxes(title_text="Index", range=[0, 100], secondary_y=True)
    return fig


def water_balance_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df["date"], y=df["ETc"], name="Crop water use ETc (mm)",
                         marker_color="#f97316"))
    fig.add_trace(go.Bar(x=df["date"], y=df["rain"], name="Rain (mm)", marker_color="#38bdf8"))
    fig.add_trace(go.Scatter(x=df["date"], y=df["deficit"], name="Deficit (mm)",
                             mode="lines+markers",
                             marker=dict(color=[ADVISORY_COLORS[int(a)] for a in df["advisory"]],
                                         size=10, line=dict(color="#111", width=1)),
                             line=dict(color="#111")))
    fig.update_layout(title="8-day water balance", barmode="group", height=320,
                      margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.2))
    fig.update_yaxes(title_text="mm per 8 days")
    return fig


def confusion_heatmap(metrics: dict) -> go.Figure:
    names = metrics["names"]
    fig = go.Figure(go.Heatmap(z=metrics["matrix"], x=names, y=names, colorscale="Greens",
                               text=metrics["matrix"], texttemplate="%{text}", showscale=False))
    fig.update_layout(title="Confusion matrix (rows = reference, columns = predicted)",
                      height=360, margin=dict(l=10, r=10, t=40, b=10))
    fig.update_yaxes(autorange="reversed")
    return fig


def importance_chart(importance: dict, top: int = 15) -> go.Figure:
    items = sorted(importance.items(), key=lambda kv: kv[1], reverse=True)[:top][::-1]
    fig = go.Figure(go.Bar(x=[v for _, v in items], y=[k for k, _ in items], orientation="h",
                           marker_color="#15803d"))
    fig.update_layout(title=f"Top {top} features (Random Forest importance)", height=420,
                      margin=dict(l=10, r=10, t=40, b=10))
    return fig


def stage_name(s) -> str:
    return STAGE_NAMES.get(int(s), "–") if pd.notna(s) else "–"


def advisory_name(a) -> str:
    return ADVISORY_LABELS.get(int(a), "–") if pd.notna(a) else "–"


def ee_geom_to_geojson(geom: ee.Geometry) -> dict:
    return geom.getInfo()
