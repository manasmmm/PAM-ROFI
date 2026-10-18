# PAM-ROFI

**Phenology-Aligned Multi-sensor Radar–Optical Fusion** for crop type mapping, moisture stress detection and 8-day irrigation advisory.

Built for Bharatiya Antariksh Hackathon **PS-06: AI-based Automated Crop Type & Moisture Stress Detection using Optical & Microwave Satellite Data**. Team JugadX, IIT Guwahati.

It uses free data only (Sentinel-1, Sentinel-2, ERA5-Land, CHIRPS, MODIS, ESA WorldCover, SMAP) and runs on Google Earth Engine. You get a web app where you pick a region and an 8-day period and see:

1. **Crop type map**: Non-agri, Sugarcane, Paddy, Other crops/fallow (Kharif) or Wheat (Rabi).
2. **Phase-aware moisture stress**: No stress, Watch, or Severe, judged against each field's own growth stage.
3. **Irrigation advisory**: No irrigation, Irrigate within 3–4 days, or Irrigate now, with a depth in mm, a field-level message in English or Hindi, and hectare totals per crop.

---

## What changed from the hackathon notebook

| Area | Hackathon notebook | This version |
|---|---|---|
| Labels & accuracy | Labels generated from the same NDVI/SAR thresholds the model learns from, so the 94% was circular | Real digitised fields (labeling tool included). Train/test split **by field**. Confusion matrix, kappa, per-class F1. Demo "weak" mode is clearly flagged UNVALIDATED |
| Classifier input | One season median | Full multi-date trajectory (NDVI, LSWI, NDRE, VV, VH, VH/VV per 10-day step) **plus per-pixel phenology features** (peak time, start of season, VH flood-dip time, amplitude, green-season length). This is "Tabular PAM" done per pixel |
| PAM stage | Month → token, same for every pixel | Each pixel's **own sowing / transplant date** from its own time series (VH minimum for paddy, NDVI start-of-season for other crops), giving its own Days-After-Sowing, stage token and Kc |
| Cloud masking | QA60 bits | Cloud Score+ (current best practice), plus gap-filling from neighbouring windows. Gap-filled points are marked in charts |
| VCI | Min/max of the current season only | Multi-year baseline for the same calendar window (Kogan VCI) |
| SMI | Season min/max VV | Multi-year dry/wet SAR references (5th/95th percentile). Script to validate against SMAP |
| Stress logic | Fixed matrix, late stage forced to 0 | Stage-aware: soil-only at the initial stage (flooded paddy is never stressed), full matrix at development/mid-season, and at late stage only joint canopy + soil collapse counts |
| ET₀ | Used MOD16 **actual ET** band by mistake | Hargreaves ET₀ from ERA5-Land (FAO-56 eq. 52), or MOD16 **PET** |
| Water balance | Season median of all periods, total rain | Chosen 8-day period, effective rain, paddy percolation loss, per-pixel Kc |
| Region | ~2 km² polygon (climate data is ~1 pixel there) | Presets up to a whole district, any district by name, or upload a GeoJSON |
| Delivery | Notebook | Web app, CLI scripts, tests, exports |

---

## Quick start

```bash
git clone https://github.com/manasmmm/PAM-ROFI.git
cd PAM-ROFI
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt

earthengine authenticate          # one time, opens a browser
# set your GEE project in config.yaml → ee_project  (or export PAMROFI_EE_PROJECT=...)

streamlit run app/streamlit_app.py
```

The first load of a region takes 20–60 seconds, because Earth Engine builds the season cube and trains the model. After that, results are cached.

> **Start with "Meerut pilot + 10 km buffer".** The original 2 km² pilot is too small for the 5–10 km climate data, and a full district is slower to load interactively.

### Using the app

- **Overview**: hectares per crop, and hectares in each advisory level per crop.
- **Crop map**: crop types plus an estimated sowing-date layer. *Model validation* shows the confusion matrix, kappa, per-class accuracy and feature importance.
- **Moisture stress**: stress map plus PAM stage, VCI and SMI layers.
- **Irrigation advisory**: advisory, deficit in mm and Kc layers.
- **Field inspector**: click any field on any map to see its NDVI/VH trajectory with growth stages shaded, its stress history, its 8-day water balance, and the farmer message in English or Hindi.
- **Export**: GeoTIFF download (small regions), Google Drive export (large regions), and a CSV summary.

---

## Getting a *real* accuracy (do this first)

1. Open <https://code.earthengine.google.com>, paste in `scripts/labeling_tool.js`, and press **Run**.
2. Click a field to see its season signature. Then draw a polygon **inside** the field in the matching layer:
   - **Sugarcane**: NDVI already high in June and stays high all season.
   - **Paddy**: VH drops sharply in June–July (flooded field), then NDVI climbs fast.
   - **Other**: one shorter green-up (maize, pulses, fodder), or flat (fallow).
   - **Non-agri**: buildings, roads, orchards, water, brick kilns.
3. Aim for **40–80 fields per class**, spread over the whole region. Split the work so each team member labels one area.
4. Press **Export labels**, then run both tasks in the *Tasks* tab.
5. Either put the asset id in `config.yaml → classification.labels`, or download the GeoJSON into `labels/` and pick it in the app's sidebar.
6. Run the evaluation:

```bash
python scripts/train_evaluate.py --roi meerut_buffer_10km
```

This prints the confusion matrix, overall accuracy, kappa, per-class producer's and user's accuracy, F1 and the top features, and saves `reports/accuracy.json`. **This is the number to put in your report.** If someone from the team can visit fields near Meerut with a phone GPS, even 50 verified points make the validation much stronger.

---

## Command-line tools

```bash
# accuracy report
python scripts/train_evaluate.py --labels labels/meerut_kharif_2025.geojson

# advisory for one 8-day period (+ farmer message for one field, + Drive export)
python scripts/run_advisory.py --date 2025-09-10 --point 77.685 29.117 --lang hi --export

# check the radar soil-moisture index against NASA SMAP
python scripts/validate_smi.py --roi meerut_buffer_10km
```

---

## Deploy online (free)

**Streamlit Community Cloud**

1. In Google Cloud, create a **service account** for your GEE project and download its JSON key. Register the service account for Earth Engine at <https://signup.earthengine.google.com/#!/service_accounts>.
2. Push this repo to GitHub. Never commit the key.
3. On <https://share.streamlit.io>, create a new app with main file `app/streamlit_app.py`.
4. In *Settings → Secrets*, paste the key as shown in `.streamlit/secrets.toml.example`.

Hugging Face Spaces (Streamlit SDK) works the same way: add the key as a secret named `EE_SERVICE_ACCOUNT_JSON`.

---

## Project layout

```
config.yaml              all regions, seasons, crop Kc curves, thresholds
pamrofi/
  config.py              config loading + validation
  agro.py                pure-Python formulas: FAO-56 Kc, PAM stages, Hargreaves,
                         water balance, stress matrix, farmer messages (unit-tested)
  data.py                Sentinel-1/2, ERA5-Land, CHIRPS, MODIS, WorldCover, SMAP loaders
  features.py            10-day gap-filled composites, temporal stack, phenology metrics
  phenology.py           per-pixel sowing date → DAS → PAM stage → Kc
  classify.py            labels, spatially-split Random Forest, accuracy metrics
  stress.py              multi-year VCI, SAR SMI, phase-aware stress
  advisory.py            8-day water balance and advisory levels
  pipeline.py            ties it all together; stats, field time series, exports
  earthengine.py         GEE login (personal or service account)
app/                     Streamlit web app
scripts/                 labeling tool (GEE JS), train/evaluate, advisory, SMI validation
tests/                   formula tests, full-graph smoke test, headless app test
docs/METHODOLOGY.md      method write-up for the report
legacy/                  original hackathon notebook
```

Run the tests with `pip install -r requirements-dev.txt && pytest` (no Earth Engine account needed).

## Configuration

Everything that can change lives in `config.yaml`:

- regions
- seasons (a Rabi 2025–26 preset with wheat is included)
- crop classes and colours
- FAO-56 Kc values and stage lengths
- the sowing-detection method for each crop
- stress thresholds
- ET₀ and rain sources
- advisory thresholds

To add a new year, copy a season block and change the dates.

## Known limitations (say these openly, judges respect it)

- The SAR soil-moisture index is a *relative* change-detection proxy. Dense canopy weakens it in mid-season, which is why it is combined with VCI and validated against SMAP.
- Rainfall (CHIRPS ~5 km, ERA5 ~9 km) and ET₀ are coarse, so the deficit varies mostly by crop and stage within a block. Plugging in IMD gridded rain or a local weather station is the natural upgrade.
- The water balance is per 8-day period with no soil-water carry-over. A FAO-56 soil-water bucket is the next step.
- CHIRPS final data lags by a few weeks. For near-real-time use, choose ERA5-Land rain in the sidebar.
