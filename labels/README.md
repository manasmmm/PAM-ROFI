# Training labels

Put digitised field labels here as GeoJSON (or keep them as a GEE asset).

Each feature needs a numeric property `class` matching the season's class ids in
`config.yaml` (kharif: 0 non-agri, 1 sugarcane, 2 paddy, 3 other/fallow).
Polygons are best (drawn inside fields, away from edges); GPS points also work.

How to make them: open `scripts/labeling_tool.js` in the GEE Code Editor, draw
fields, press **Export labels**, then either
* use the asset id in `config.yaml → classification.labels`, or
* download the GeoJSON from Drive into this folder and select it in the app.

Target: 40–80 fields per class spread across the whole region. Train/test is
split by field, so each field is either training or testing, never both.
