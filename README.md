# PAM-ROFI: Phenology-Aligned Multi-Sensor Irrigation Advisory Engine

A cloud-native, spatial-temporal pipeline deployed on Google Earth Engine (GEE) to dynamically map crop distributions, monitor phase-aware vegetative moisture stress, and compute regional 8-day command-scale irrigation advisories over Meerut, Western UP.

---

## 🛰️ Core System Objectives

### 1. Regional Land-Use Crop Type Mapping
- **Multi-Sensor Fusion:** Ingests and registers co-polarized Sentinel-1 GRD (C-band SAR) and Sentinel-2 Surface Reflectance (SR) optical streams into a unified feature stack to eliminate cloud gaps.
- **Physics-Backed Auto-Labeling:** Avoids manual ground-truth vulnerabilities by leveraging long-term temporal percentile profiles (NDVI max bounds and SAR VV backscatter delta) to automatically classify 5,000 sandbox feature coordinates.
- **High-Performance Classification:** Executes a 100-tree Random Forest classifier optimized via `.sampleRegions()` array metrics to segment the pilot region into four strict categories: *Non-Agri (Class 0)*, *Sugarcane (Class 1)*, *Paddy Rice (Class 2)*, and *Wheat/Seasonal Fallow (Class 3)*.

### 💧 2. Phase-Aware Crop Moisture Stress Tracking
- **The PAM Paradigm:** Incorporates a custom **Phenology Alignment Method (PAM)** engine that directly injects discrete temporal metadata tokens (stages 0–3) into the tabular feature matrix, eliminating the need for computationally heavy recurrent deep learning frameworks.
- **Lifecycle-Grounded Context:** Normalizes local anomalies like delayed sowing or climate shocks by grounding moisture detection in true physiological timelines instead of blind, rigid calendar months.
- **Dual-Indicator Matrix:** Cross-references the Vegetation Condition Index (VCI) with the Soil Moisture Index (SMI) to distinguish standard physiological pre-harvest drawdowns from destructive root-zone hydrological drought.

### 🚜 3. Hydrological Deficit Modeling & Advisory Tiering
- **Dynamic Demand Compounding:** Calculates real-time daily crop water consumption ($ET_c = ET_0 \times K_c$) by scaling 8-day MOD16A2 potential evapotranspiration maps against continuous, piecewise-linear FAO-56 crop coefficient ($K_c$) curves driven by PAM Days After Sowing (DAS) tokens.
- **Mass Balance Computation:** Resolves a pixel-level mass-balance equation ($\text{Deficit} = ET_c - \text{Rainfall}$) by balancing water demand against high-resolution CHIRPS daily precipitation layers upscaled using bicubic interpolation.
- **Advisory Classification:** Floors negative depths to `max(0)` and partitions active deficit results into three explicit, actionable emergency polygons:
  - **Priority 0 (< 10mm):** No Irrigation Needed (Green)
  - **Priority 1 (10–30mm):** Routine Irrigation Schedule within 3–4 Days (Orange)
  - **Priority 2 (> 30mm):** Critical Deficit — Immediate Irrigation Required (Red)

---

## 💻 Technical Stack
- **Cloud Computing Engine:** Google Earth Engine (GEE) Ecosystem (`ee`)
- **Interactive Visualization Panel:** `geemap`, `ipyleaflet`, `ipywidgets`
- **Primary Datasets:** Sentinel-1 GRD, Sentinel-2 SR, MODIS (MOD16A2GF), CHIRPS Daily

---

## 🚀 Native Local Deployment

1. Clone the repository to your environment:
   ```bash
   git clone [https://github.com/manasmmm/PAM-ROFI.git](https://github.com/manasmmm/PAM-ROFI.git)
