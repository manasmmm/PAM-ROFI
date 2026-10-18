"""PAM-ROFI web app.

Run locally:   streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import json
import math
import os
import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import ee  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from streamlit_folium import st_folium  # noqa: E402

import components as ui  # noqa: E402
from pamrofi import data as eedata  # noqa: E402
from pamrofi import earthengine, viz  # noqa: E402
from pamrofi.agro import ADVISORY_COLORS, ADVISORY_LABELS, FieldAdvice  # noqa: E402
from pamrofi.config import Config  # noqa: E402
from pamrofi.pipeline import Pipeline, PipelineOptions  # noqa: E402

st.set_page_config(page_title="PAM-ROFI · Crop, Stress & Irrigation Advisory",
                   page_icon="🛰️", layout="wide")


# ---------------------------------------------------------------- setup
@st.cache_resource(show_spinner=False)
def init_ee(project: str | None) -> None:
    sa = None
    try:
        sa = st.secrets.get("EE_SERVICE_ACCOUNT_JSON")  # Streamlit Cloud secret
    except Exception:  # noqa: BLE001 – no secrets file locally
        sa = None
    if isinstance(sa, dict):
        sa = json.dumps(dict(sa))
    earthengine.init(project=project, service_account_json=sa, interactive=False)


@st.cache_resource(show_spinner=False)
def load_config() -> Config:
    return Config.load(os.environ.get("PAMROFI_CONFIG"))


@st.cache_resource(show_spinner="Building the season data cube and training the crop model…")
def get_pipeline(season_key: str, roi_key: str, roi_geojson: str | None, district: str | None,
                 labels: str, et0_source: str, rain_source: str, th_low: float, th_high: float,
                 orbit: str | None) -> Pipeline:
    cfg = load_config().with_overrides(
        classification={"labels": labels},
        advisory={"et0_source": et0_source, "rain_source": rain_source,
                  "thresholds_mm": [th_low, th_high]},
    )
    geom = None
    if roi_geojson:
        geom = eedata.roi_from_geojson(roi_geojson)
    elif district:
        name, _, state = district.partition(",")
        geom = eedata.district_geometry(name.strip(), state.strip() or None)
    return Pipeline(cfg, PipelineOptions(season_key=season_key, roi_key=roi_key,
                                         roi_geometry=geom, orbit_pass=orbit))


@st.cache_data(show_spinner=False, ttl=3600)
def tile_url(key: str, _image: ee.Image, vis: dict) -> str:
    return _image.getMapId(vis)["tile_fetcher"].url_format


@st.cache_data(show_spinner=False, ttl=3600)
def roi_view(key: str, _pipe: Pipeline) -> tuple[dict, tuple[float, float], int]:
    info = ee.Dictionary({"geo": _pipe.roi, "area": _pipe.roi.area(10),
                          "c": _pipe.roi.centroid(10).coordinates()}).getInfo()
    km2 = info["area"] / 1e6
    zoom = int(max(9, min(16, 14 - math.log2(max(km2, 1)) / 2)))
    lon, lat = info["c"]
    return info["geo"], (lat, lon), zoom


@st.cache_data(show_spinner="Reading the field's full season from the satellites…", ttl=3600)
def field_series(key: str, _pipe: Pipeline, lon: float, lat: float) -> dict:
    return _pipe.point_timeseries(lon, lat)


@st.cache_data(show_spinner="Computing area statistics…", ttl=3600)
def area_stats(key: str, _pipe: Pipeline) -> dict:
    return _pipe.area_by_class()


@st.cache_data(show_spinner="Summarising the advisory…", ttl=3600)
def adv_summary(key: str, _pipe: Pipeline, d: str) -> list[dict]:
    return _pipe.advisory_summary(d)


@st.cache_data(show_spinner="Validating the crop model on held-out fields…", ttl=3600)
def validation(key: str, _pipe: Pipeline) -> dict:
    m = _pipe.evaluate()
    return {"metrics": m, "importance": _pipe.train_result.importance,
            "n_train": _pipe.train_result.n_train, "n_test": _pipe.train_result.n_test}


def render_map(key: str, layers: list[tuple[str, ee.Image, dict, bool]], legend: tuple[str, dict],
               height: int = 560):
    geo, center, zoom = roi_view(PKEY, pipe)
    m = ui.base_map(center, zoom)
    for name, img, vis, shown in layers:
        ui.add_ee_layer(m, tile_url(f"{PKEY}|{key}|{name}", img, vis), name, shown)
    ui.add_roi_outline(m, geo)
    if st.session_state.get("field"):
        lat, lon = st.session_state["field"]
        ui.add_marker(m, lat, lon)
    ui.add_legend(m, *legend)
    out = st_folium(ui.finish(m), height=height, use_container_width=True, key=f"map_{key}",
                    returned_objects=["last_clicked"])
    if out and out.get("last_clicked"):
        clicked = (round(out["last_clicked"]["lat"], 6), round(out["last_clicked"]["lng"], 6))
        if clicked != st.session_state.get("field"):
            st.session_state["field"] = clicked
            st.rerun()


# --------------------------------------------------------------- sidebar
cfg = load_config()
try:
    init_ee(cfg.ee_project)
except Exception as exc:  # noqa: BLE001
    st.error("Could not connect to Google Earth Engine.")
    st.markdown("Run `earthengine authenticate` once on this machine, or add a service-account "
                "key as the secret `EE_SERVICE_ACCOUNT_JSON` (see README).")
    st.exception(exc)
    st.stop()

with st.sidebar:
    st.title("🛰️ PAM-ROFI")
    st.caption("Phenology-aligned radar–optical fusion for crop type, moisture stress "
               "and irrigation advisory")

    season_key = st.selectbox("Season", cfg.season_keys,
                              format_func=lambda k: cfg.season(k).label)
    season = cfg.season(season_key)

    roi_mode = st.radio("Region", ["Preset", "District name", "Upload GeoJSON"], horizontal=True)
    roi_key, roi_geojson, district = cfg.raw["default_roi"], None, None
    if roi_mode == "Preset":
        roi_key = st.selectbox("Preset region", list(cfg.rois),
                               index=list(cfg.rois).index(cfg.raw["default_roi"]),
                               format_func=lambda k: cfg.rois[k].get("label", k))
    elif roi_mode == "District name":
        district = st.text_input("District, State (FAO GAUL names)", "Meerut, Uttar Pradesh")
    else:
        up = st.file_uploader("GeoJSON polygon", type=["geojson", "json"])
        if up:
            roi_geojson = up.read().decode("utf-8")

    with st.expander("Model & data options"):
        label_files = sorted(str(p.relative_to(ROOT)) for p in (ROOT / "labels").glob("*.geojson"))
        label_choice = st.selectbox(
            "Training labels", ["weak (demo, unvalidated)"] + label_files + ["GEE asset…"])
        if label_choice.startswith("weak"):
            labels = "weak"
        elif label_choice == "GEE asset…":
            labels = st.text_input("Asset id", "projects/<project>/assets/pamrofi_labels") or "weak"
        else:
            labels = label_choice
        et0_source = st.selectbox("ET₀ source", ["hargreaves_era5", "modis_pet"],
                                  format_func={"hargreaves_era5": "Hargreaves (ERA5-Land)",
                                               "modis_pet": "MODIS MOD16 PET"}.get)
        rain_source = st.selectbox("Rainfall source", ["chirps", "era5"],
                                   format_func={"chirps": "CHIRPS", "era5": "ERA5-Land"}.get)
        th = cfg.section("advisory").get("thresholds_mm", [10, 30])
        th_low, th_high = st.slider("Advisory thresholds (mm deficit)", 0, 80,
                                    (int(th[0]), int(th[1])))
        orbit = st.selectbox("Sentinel-1 orbit", ["Both", "ASCENDING", "DESCENDING"])
        orbit = None if orbit == "Both" else orbit

    adv_cfg = cfg.section("advisory")
    periods = ui.period_starts(season.start, season.end, int(adv_cfg.get("period_days", 8)))
    latest_ok = date.today() - timedelta(days=10)  # climate data latency
    default_idx = max(i for i, p in enumerate(periods) if p <= latest_ok) if periods[0] <= latest_ok else 0
    period = st.select_slider("Evaluation period (8 days starting)", options=periods,
                              value=periods[min(default_idx, len(periods) - 1)],
                              format_func=lambda d: d.strftime("%d %b %Y"))
    lang = st.radio("Farmer message language", ["English", "हिन्दी"], horizontal=True)

PKEY = "|".join(map(str, [season_key, roi_key, hash(roi_geojson), district, labels, et0_source,
                          rain_source, th_low, th_high, orbit]))
try:
    pipe = get_pipeline(season_key, roi_key, roi_geojson, district, labels, et0_source,
                        rain_source, float(th_low), float(th_high), orbit)
except Exception as exc:  # noqa: BLE001
    st.error("Could not build the pipeline for this region/season.")
    st.exception(exc)
    st.stop()

period_str = period.isoformat()
period_end = (period + timedelta(days=int(adv_cfg.get("period_days", 8)) - 1)).isoformat()

# ---------------------------------------------------------------- header
st.markdown(f"## {season.label} · advisory for **{period:%d %b} – {date.fromisoformat(period_end):%d %b %Y}**")
if not pipe.validated:
    st.warning("**Demo mode — crop map is UNVALIDATED.** It is trained on rule-generated "
               "(weak) labels, so any accuracy you compute only measures agreement with the "
               "rules. Digitise real fields with `scripts/labeling_tool.js` and select them under "
               "*Model & data options* to get a real accuracy.", icon="⚠️")

tabs = st.tabs(["📊 Overview", "🌾 Crop map", "💧 Moisture stress", "🚜 Irrigation advisory",
                "📍 Field inspector", "⬇️ Export"])

# -------------------------------------------------------------- overview
with tabs[0]:
    try:
        areas = area_stats(PKEY, pipe)
        rows = adv_summary(PKEY, pipe, period_str)
    except Exception as exc:  # noqa: BLE001
        st.error("Statistics failed (often a region that is too large for interactive use).")
        st.exception(exc)
        areas, rows = {}, []
    if areas:
        cols = st.columns(len(areas))
        for col, (name, ha) in zip(cols, areas.items()):
            col.metric(name, f"{ha:,.0f} ha")
    if rows:
        df = pd.DataFrame(rows)
        df["advisory"] = df["advisory"].map(ADVISORY_LABELS)
        tot = df.groupby("advisory")["hectares"].sum()
        c1, c2, c3 = st.columns(3)
        for col, lvl in zip((c1, c2, c3), range(3)):
            col.markdown(
                f'<div style="border-left:6px solid {ADVISORY_COLORS[lvl]};padding:6px 12px">'
                f'<div style="font-size:.85rem;opacity:.8">{ADVISORY_LABELS[lvl]}</div>'
                f'<div style="font-size:1.6rem;font-weight:700">'
                f'{tot.get(ADVISORY_LABELS[lvl], 0):,.0f} ha</div></div>',
                unsafe_allow_html=True)
        st.markdown("#### Advisory by crop")
        st.dataframe(df.pivot_table(index="crop", columns="advisory", values="hectares",
                                    aggfunc="sum", fill_value=0).round(1),
                     width="stretch")
    st.caption("Click any map to pick a field, then open **Field inspector**.")

# -------------------------------------------------------------- crop map
with tabs[1]:
    sowing = pipe.sowing_doy()
    render_map("crop", [
        ("Crop type", pipe.crop_map, viz.crop_vis(season), True),
        ("Estimated sowing / transplant day-of-year", sowing, viz.SOWING_VIS, False),
    ], ("Crop type", viz.crop_legend(season)))
    with st.expander("Model validation", expanded=pipe.validated):
        if st.button("Run validation on held-out fields"):
            st.session_state["val"] = validation(PKEY, pipe)
        v = st.session_state.get("val")
        if v:
            m = v["metrics"]
            if m["overall_accuracy"] is None:
                st.info("No test samples in this region.")
            else:
                note = "" if pipe.validated else " (agreement with weak rules, not accuracy)"
                a, b, c = st.columns(3)
                a.metric("Overall accuracy" + note, f"{m['overall_accuracy']*100:.1f}%")
                b.metric("Cohen's kappa", f"{m['kappa']:.3f}")
                c.metric("Train / test samples", f"{v['n_train']} / {v['n_test']}")
                st.plotly_chart(ui.confusion_heatmap(m), width="stretch")
                st.dataframe(pd.DataFrame(m["per_class"]).T.round(3), width="stretch")
                if v["importance"]:
                    st.plotly_chart(ui.importance_chart(v["importance"]), width="stretch")

# --------------------------------------------------------------- stress
with tabs[2]:
    st_img = pipe.stress(period_str)
    render_map("stress", [
        ("Moisture stress", st_img.select("stress"), viz.STRESS_VIS, True),
        ("PAM growth stage", st_img.select("PAM_stage").selfMask(), viz.STAGE_VIS, False),
        ("VCI (canopy condition)", st_img.select("VCI"), viz.VCI_VIS, False),
        ("SMI (radar soil moisture)", st_img.select("SMI"), viz.SMI_VIS, False),
    ], ("Moisture stress", viz.STRESS_LEGEND))
    st.caption("Stress logic is stage-aware: at the initial stage only radar soil moisture "
               "counts (flooded paddy is never stressed); at development/mid-season both canopy "
               "(VCI) and soil (SMI) count; at late season only a joint canopy + soil collapse "
               "is flagged, because falling greenness is normal senescence.")

# ------------------------------------------------------------- advisory
with tabs[3]:
    adv = pipe.advisory(period_str)
    render_map("advisory", [
        ("Irrigation advisory", adv.select("advisory"), viz.ADVISORY_VIS, True),
        ("Water deficit (mm / 8 days)", adv.select("deficit"), viz.DEFICIT_VIS, False),
        ("Crop coefficient Kc", adv.select("Kc"), {"min": 0, "max": 1.3,
                                                    "palette": ["#fff7ed", "#ea580c"]}, False),
    ], ("Irrigation advisory", viz.ADVISORY_LEGEND))
    st.caption("Deficit = ET₀ × Kc (+ paddy percolation) − effective rain, per pixel, with Kc "
               "from each field's own sowing date.")

# ------------------------------------------------------- field inspector
with tabs[4]:
    c1, c2, c3 = st.columns([1, 1, 1])
    f = st.session_state.get("field")
    lat_in = c1.number_input("Latitude", value=float(f[0]) if f else 29.117, format="%.6f")
    lon_in = c2.number_input("Longitude", value=float(f[1]) if f else 77.685, format="%.6f")
    if c3.button("Inspect this field", width="stretch"):
        st.session_state["field"] = (lat_in, lon_in)
        f = st.session_state["field"]
    if not f:
        st.info("Click a field on any map, or enter coordinates and press **Inspect**.")
    else:
        try:
            ts = field_series(PKEY, pipe, f[1], f[0])
        except Exception as exc:  # noqa: BLE001
            st.error("Could not read this location.")
            st.exception(exc)
            ts = None
        if ts:
            stat = ts["static"] or {}
            cls = stat.get("class")
            crop = season.class_names.get(int(cls), "Unknown") if cls is not None else "Outside ROI"
            comp = pd.DataFrame(ts["composites"])
            advdf = pd.DataFrame(ts["advisory"])
            for df_ in (comp, advdf):
                if not df_.empty:
                    df_["date"] = pd.to_datetime(df_["date"])
            sow_t = stat.get("sowing_t")
            sow_txt = "–"
            if sow_t is not None and sow_t > -9000 and stat.get("active"):
                sow_txt = (date.fromisoformat(ts["season_start"]) +
                           timedelta(days=int(sow_t))).strftime("%d %b %Y")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Crop", crop)
            m2.metric("Estimated sowing / transplant", sow_txt)
            m3.metric("Peak NDVI", f"{stat.get('ndvi_max', 0) or 0:.2f}")
            m4.metric("NDVI amplitude", f"{stat.get('ndvi_amp', 0) or 0:.2f}")

            cur = advdf[advdf["date"] == pd.Timestamp(period)] if not advdf.empty else advdf
            if stat.get("active") and not cur.empty:
                row = cur.iloc[0]
                cur_stage = comp.loc[comp["date"] <= pd.Timestamp(period), "PAM_stage"]
                stage = int(cur_stage.iloc[-1]) if len(cur_stage) and pd.notna(cur_stage.iloc[-1]) else 0
                advice = FieldAdvice(crop=crop, stage=stage, deficit_mm=float(row["deficit"]),
                                     level=int(row["advisory"]), period_start=period_str,
                                     period_end=period_end)
                st.markdown(
                    f'<div style="border-left:8px solid {ADVISORY_COLORS[advice.level]};'
                    f'padding:10px 14px;background:rgba(127,127,127,.08);border-radius:6px;'
                    f'font-size:1.05rem">{advice.message("hi" if lang != "English" else "en")}'
                    f'</div>', unsafe_allow_html=True)
            elif not stat.get("active"):
                st.info("This pixel is non-agricultural or fallow this season — no advisory.")

            if not comp.empty:
                st.plotly_chart(ui.trajectory_chart(comp), width="stretch")
                st.plotly_chart(ui.stress_chart(comp), width="stretch")
            if not advdf.empty and stat.get("active"):
                st.plotly_chart(ui.water_balance_chart(advdf), width="stretch")
            with st.expander("Raw values"):
                st.dataframe(comp, width="stretch")
                st.dataframe(advdf, width="stretch")

# --------------------------------------------------------------- export
with tabs[5]:
    st.markdown("Direct GeoTIFF downloads work for small regions (< ~32 MB). For a district, "
                "use **Export to Google Drive**.")
    products = {
        "crop_map": pipe.crop_map,
        "moisture_stress": pipe.stress(period_str).select(["stress", "PAM_stage", "VCI", "SMI"]),
        "irrigation_advisory": pipe.advisory(period_str).select(["advisory", "deficit", "ETc",
                                                                 "ET0", "rain", "Kc"]),
        "sowing_doy": pipe.sowing_doy(),
    }
    choice = st.selectbox("Product", list(products))
    scale = st.select_slider("Resolution (m)", [10, 20, 30, 60], value=10)
    a, b = st.columns(2)
    if a.button("Get download link"):
        try:
            url = pipe.download_url(products[choice], scale=scale, name=f"pamrofi_{choice}")
            st.markdown(f"[⬇️ Download {choice}.tif]({url})")
        except Exception as exc:  # noqa: BLE001
            st.error("Too large for a direct download — use Drive export or a coarser resolution.")
            st.caption(str(exc))
    if b.button("Export to Google Drive"):
        task = pipe.export_image(products[choice], f"pamrofi_{choice}_{period_str}", scale=scale)
        st.success(f"Export started (task {task.id}). Check the Tasks tab in the GEE Code Editor "
                   "or your Drive folder PAM_ROFI.")
    try:
        rows = adv_summary(PKEY, pipe, period_str)
        if rows:
            csv = pd.DataFrame(rows).assign(
                advisory_label=lambda d: d["advisory"].map(ADVISORY_LABELS)).to_csv(index=False)
            st.download_button("Download advisory summary (CSV)", csv,
                               file_name=f"advisory_{period_str}.csv", mime="text/csv")
    except Exception:  # noqa: BLE001
        pass
