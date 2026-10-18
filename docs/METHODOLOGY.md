# PAM-ROFI — Methodology

This document is a ready-to-adapt methodology section for the project report.

## 1. Study area and period
Western Uttar Pradesh (Meerut). The default demo region is a 10 km buffer around the original pilot block. Kharif 2025 runs 1 Jun – 30 Nov 2025. A Rabi 2025–26 configuration is included.

## 2. Data
| Dataset | Use | Resolution |
|---|---|---|
| Sentinel-2 L2A (harmonized) + Cloud Score+ | NDVI, EVI, NDWI, LSWI, NDRE | 10–20 m, 5 days |
| Sentinel-1 GRD IW (VV, VH) | backscatter, flood signal, soil moisture proxy | 10 m, 6–12 days |
| ERA5-Land daily | Tmax/Tmin → Hargreaves ET₀; optional rain | ~9 km |
| CHIRPS daily | rainfall | ~5 km |
| MOD16A2GF | alternative ET₀ (PET band) | 500 m, 8 days |
| ESA WorldCover 2021 | non-agricultural mask, weak labels | 10 m |
| SMAP L4 | validation of the SAR soil-moisture index | 9 km |

## 3. Pre-processing
* **Optical.** Pixels with Cloud Score+ `cs_cdf` < 0.6 are masked. Reflectance is scaled to 0–1 and indices are computed.
* **SAR.** Values below −30 dB are masked as border noise. A 30 m focal-median speckle filter is applied, and the VH/VV ratio is computed as a dB difference.
* **Regular composites.** The season is cut into 10-day windows and each window takes the median of its images. If a window is empty, it is filled from a ±10-day window, and then from the season median. A per-window flag (`obs`) records whether a real cloud-free observation existed.

## 4. Crop type classification (Objective 1)
**Features (per pixel).**
* The full temporal stack: NDVI, LSWI, NDRE, VV, VH and VH/VV for each 10-day composite.
* **Phenology features (Tabular PAM).**
  * Peak NDVI and its timing
  * Minimum NDVI and amplitude
  * Start of season: first time NDVI exceeds min + 35% of amplitude
  * Timing and depth of the VH minimum (paddy flooding)
  * VH range
  * Mean LSWI
  * Green-season length

Encoding *when* phenological events happen as tabular features gives the Random Forest the temporal awareness that sequence models get from recurrence or attention, at a fraction of the cost.

**Labels.** Field polygons are digitised in the GEE labeling tool, guided by each field's NDVI/VH signature and very-high-resolution imagery. Up to 25 points are sampled inside each polygon, after shrinking it by 10 m to avoid edge pixels.

**Validation.** 30% of **polygons** are held out. All points from a held-out field are excluded from training, which avoids the spatial autocorrelation leakage of pixel-level random splits. We report the confusion matrix, overall accuracy, Cohen's κ, producer's and user's accuracy, and F1 per class.

**Model.** Random Forest, 200 trees (GEE `smileRandomForest`), with Gini feature importance.

## 5. Phenology alignment (PAM)
Each crop pixel gets its own sowing date:
* **Paddy.** The transplanting date is the date of the VH backscatter minimum within 10 Jun – 31 Aug, when the flooded field gives specular reflection.
* **Seasonal crops.** Start of season from the NDVI threshold method, minus an emergence lag of 10–12 days.
* **Sugarcane** (planted before the window). A configured planting date (Feb/Mar).

Days-After-Sowing (DAS) on any date gives the PAM token for that pixel:

| Token | Meaning |
|---|---|
| 0 | out of season |
| 1 | initial stage |
| 2 | development + mid-season (water-critical) |
| 3 | late season |

DAS also gives the FAO-56 four-stage piecewise-linear crop coefficient:

| Stage | Kc |
|---|---|
| Initial | Kc_ini |
| Development | linear from Kc_ini to Kc_mid |
| Mid-season | Kc_mid |
| Late season | linear from Kc_mid to Kc_end |

Stage lengths and Kc values follow FAO-56 Tables 11–12, adjusted to typical Western UP varieties. Class 3 pixels with an NDVI amplitude below 0.2 are treated as fallow and receive no advisory.

## 6. Moisture stress (Objective 2)
* **VCI** = 100 · (NDVI − NDVI_min) / (NDVI_max − NDVI_min). Min and max are taken over the same ±15-day calendar window in the current year and the previous four years (Kogan 1995).
* **SMI** = (VV − VV_dry) / (VV_wet − VV_dry). VV_dry and VV_wet are the 5th and 95th percentiles of VV over the same season in the current and previous four years. This is a change-detection soil-moisture proxy (Wagner et al. 1999).
* **Phase-aware decision matrix:**

| PAM stage | Severe (2) | Watch (1) | Note |
|---|---|---|---|
| Initial | SMI < 0.35 | SMI < 0.50 | VCI ignored (canopy not established); flooded paddy (LSWI > 0.1) = 0 |
| Development / mid | VCI < 35 **or** SMI < 0.35 | VCI < 60 **and** SMI < 0.50 | most water-sensitive stage |
| Late | – | VCI < 35 **and** SMI < 0.35 | falling NDVI alone is senescence, not drought |

* **Cloud resilience.** If the 8-day window has no clear optical observation, a 30-day window is used, pixel by pixel. SAR needs no fallback.
* **Validation.** The ROI-mean SMI over WorldCover cropland is correlated with SMAP L4 surface soil moisture for each 8-day window (`scripts/validate_smi.py`).

## 7. Water balance and irrigation advisory (Objective 3)
For each 8-day period:
* **ET₀**: the sum of daily Hargreaves values, ET₀ = 0.0023 · (T_mean + 17.8) · √(T_max − T_min) · R_a · 0.408, with R_a from FAO-56 eq. 21. MOD16 PET can be used instead.
* **ETc** = ET₀ · Kc(DAS at the period midpoint).
* **Deficit** = max(0, ETc + percolation − 0.8 · rain). Percolation is 3 mm/day for paddy while it is in the field.
* **Advisory**:

| Deficit | Advisory |
|---|---|
| < 10 mm | 0, no irrigation |
| 10–30 mm | 1, irrigate within 3–4 days |
| > 30 mm | 2, irrigate immediately |

The advisory is delivered as a map, as hectares per crop and level, and as a per-field message in English or Hindi with the recommended depth.

## 8. Verification
* Every formula in `pamrofi/agro.py` is unit-tested. For example, R_a reproduces FAO-56 Example 8 (32.2 MJ m⁻² day⁻¹).
* A mocked-Earth-Engine smoke test builds the complete computation graph for every season, region, label source and ET₀ option.
* The web app is tested headlessly.

## References
* Allen R.G. et al. (1998). *FAO Irrigation and Drainage Paper 56: Crop evapotranspiration.*
* Hargreaves G.H., Samani Z.A. (1985). Reference crop evapotranspiration from temperature. *Appl. Eng. Agric.*
* Kogan F.N. (1995). Application of vegetation index and brightness temperature for drought detection. *Adv. Space Res.*
* Wagner W. et al. (1999). A method for estimating soil moisture from ERS scatterometer and soil data. *Remote Sens. Environ.*
* Pasquarella V.J. et al. (2023). Demystifying image-based cloud masking: Cloud Score+. *CVPR Workshops.*
* Breiman L. (2001). Random Forests. *Machine Learning.*
