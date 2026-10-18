/******************************************************************************
 * PAM-ROFI field labeling tool — paste into https://code.earthengine.google.com
 *
 * What it does
 *   • Shows Sentinel-2 imagery for chosen dates plus a Sentinel-1 VH layer.
 *   • Click any field → NDVI and VH time-series chart for the whole season,
 *     so you can see the crop's signature before you label it:
 *        Sugarcane : NDVI already high in June, stays high all season
 *        Paddy     : VH drops sharply in Jun–Jul (flooded field), then NDVI rises fast
 *        Other     : single, shorter green-up (maize, pulses, fodder) or flat (fallow)
 *        Non-agri  : roads, buildings, orchards/trees, water, brick kilns
 *   • You draw polygons inside clearly identifiable fields with the drawing tools.
 *   • "Export labels" saves them as an Earth Engine asset AND a GeoJSON on Drive.
 *
 * How to label (aim for 40–80 fields per class, spread over the whole area)
 *   1. In the Geometry Imports panel (top-left of the map) you will see 4 layers
 *      created by this script: nonagri, sugarcane, paddy, other.
 *   2. Select a layer, click the polygon tool, draw INSIDE a field (stay off edges).
 *   3. Use the chart to confirm the class. When in doubt, skip that field.
 *   4. Press "Export labels" in the panel and run the tasks in the Tasks tab.
 *   5. Put the asset id (or the downloaded GeoJSON in labels/) in config.yaml
 *      → classification.labels.
 ******************************************************************************/

// ----------------------------- settings -------------------------------------
var SEASON_START = '2025-06-01';
var SEASON_END   = '2025-11-30';
var CENTER       = [77.685, 29.117];        // Meerut pilot
var ASSET_NAME   = 'pamrofi_labels_kharif2025';
var CLASSES = [
  {id: 0, name: 'nonagri',   color: '5b5b5b'},
  {id: 1, name: 'sugarcane', color: '1b7f3b'},
  {id: 2, name: 'paddy',     color: '2563eb'},
  {id: 3, name: 'other',     color: 'f59e0b'}
];

// ----------------------------- data -----------------------------------------
var region = ee.Geometry.Point(CENTER).buffer(15000).bounds();
var csPlus = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED');

var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(region).filterDate(SEASON_START, SEASON_END)
  .linkCollection(csPlus, ['cs_cdf'])
  .map(function (img) {
    var clear = img.select('cs_cdf').gte(0.6);
    var r = img.select(['B2', 'B3', 'B4', 'B8', 'B11']).divide(10000).updateMask(clear);
    return r.addBands(r.normalizedDifference(['B8', 'B4']).rename('NDVI'))
            .copyProperties(img, ['system:time_start']);
  });

var s1 = ee.ImageCollection('COPERNICUS/S1_GRD')
  .filterBounds(region).filterDate(SEASON_START, SEASON_END)
  .filter(ee.Filter.eq('instrumentMode', 'IW'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
  .select('VH');

Map.centerObject(ee.Geometry.Point(CENTER), 14);
Map.setOptions('SATELLITE');

function addMonth(m, label, show) {
  var start = ee.Date(SEASON_START).update(null, m, 1);
  var img = s2.filterDate(start, start.advance(1, 'month')).median();
  Map.addLayer(img, {bands: ['B8', 'B11', 'B4'], min: 0.02, max: 0.45}, 'S2 false colour ' + label, show);
}
addMonth(7, 'Jul', false);
addMonth(9, 'Sep', true);
addMonth(10, 'Oct', false);
Map.addLayer(s1.filterDate('2025-06-15', '2025-07-31').min(),
             {min: -25, max: -12}, 'S1 VH minimum Jun–Jul (dark = flooded paddy)', false);

// ----------------------------- drawing layers -------------------------------
var tools = Map.drawingTools();
tools.setShown(true);
tools.setDrawModes(['polygon', 'point']);
var existing = tools.layers().getJsArray().map(function (l) { return l.getName(); });
CLASSES.forEach(function (c) {
  if (existing.indexOf(c.name) === -1) {
    tools.addLayer([], c.name, '#' + c.color);
  }
});

// ----------------------------- chart panel ----------------------------------
var panel = ui.Panel({style: {width: '420px'}});
panel.add(ui.Label('PAM-ROFI labeling tool', {fontWeight: 'bold', fontSize: '18px'}));
panel.add(ui.Label('Click a field to see its season signature. Draw polygons in the ' +
                   'matching layer (nonagri / sugarcane / paddy / other).'));
var chartBox = ui.Panel();
panel.add(chartBox);

Map.onClick(function (coords) {
  var pt = ee.Geometry.Point([coords.lon, coords.lat]);
  chartBox.clear();
  chartBox.add(ui.Label('Lat ' + coords.lat.toFixed(5) + ', Lon ' + coords.lon.toFixed(5)));
  chartBox.add(ui.Chart.image.series(s2.select('NDVI'), pt, ee.Reducer.mean(), 10)
    .setOptions({title: 'NDVI', vAxis: {viewWindow: {min: 0, max: 1}}, lineWidth: 1, pointSize: 4,
                 colors: ['#15803d']}));
  chartBox.add(ui.Chart.image.series(s1, pt, ee.Reducer.mean(), 10)
    .setOptions({title: 'Sentinel-1 VH (dB)', lineWidth: 1, pointSize: 4, colors: ['#2563eb']}));
});

// ----------------------------- export ---------------------------------------
var exportBtn = ui.Button('Export labels', function () {
  var feats = [];
  tools.layers().forEach(function (layer) {
    var name = layer.getName();
    var cls = CLASSES.filter(function (c) { return c.name === name; })[0];
    if (!cls) return;
    layer.geometries().forEach(function (g) {
      feats.push(ee.Feature(g, {'class': cls.id, 'class_name': cls.name}));
    });
  });
  var fc = ee.FeatureCollection(feats);
  print('Labels by class', fc.aggregate_histogram('class_name'));
  Export.table.toAsset({collection: fc, description: ASSET_NAME, assetId: ASSET_NAME});
  Export.table.toDrive({collection: fc, description: ASSET_NAME + '_geojson',
                        fileFormat: 'GeoJSON', folder: 'PAM_ROFI'});
  print('Two export tasks created — open the Tasks tab and press RUN on both.');
});
panel.add(exportBtn);
ui.root.insert(0, panel);
