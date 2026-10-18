"""Run the Streamlit app headlessly with a fake pipeline (no Earth Engine)."""
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402


APP = str(Path(__file__).resolve().parent.parent / "app" / "streamlit_app.py")


def fake_image():
    img = mock.MagicMock(name="image")
    img.getMapId.return_value = {"tile_fetcher": SimpleNamespace(url_format="https://t/{z}/{x}/{y}")}
    img.select.return_value = img
    img.selfMask.return_value = img
    return img


class FakePipeline:
    def __init__(self, cfg, options):
        self.season = cfg.season(options.season_key)
        self.roi = mock.MagicMock()
        self.crop_map = fake_image()
        self.validated = cfg.section("classification")["labels"] != "weak"
        self.train_result = SimpleNamespace(importance={"ndvi_max": 30.0, "t_vh_min": 22.0},
                                            n_train=700, n_test=300)
        self.advisory_model = SimpleNamespace(days=8)

    def stress(self, d): return fake_image()
    def advisory(self, d): return fake_image()
    def sowing_doy(self): return fake_image()
    def area_by_class(self): return {c.name: 100.0 * (c.id + 1) for c in self.season.classes}
    def advisory_summary(self, d):
        return [{"crop": "Paddy (rice)", "advisory": 2, "hectares": 120.0, "mean_deficit_mm": 35.0},
                {"crop": "Sugarcane", "advisory": 0, "hectares": 80.0, "mean_deficit_mm": 4.0}]
    def evaluate(self):
        return {"overall_accuracy": 0.9, "kappa": 0.85, "matrix": [[9, 1], [0, 10]],
                "names": ["a", "b"], "per_class": {"a": {"f1": 0.9}, "b": {"f1": 0.95}}}
    def download_url(self, *a, **k): return "https://example.com/x.tif"
    def export_image(self, *a, **k): return SimpleNamespace(id="T1")
    def point_timeseries(self, lon, lat):
        comps = [{"date": f"2025-{m:02d}-{d:02d}", "NDVI": 0.3 + 0.05 * i, "LSWI": 0.1, "VV": -10,
                  "VH": -17 - (i % 3), "obs": i % 2, "stress": i % 3, "PAM_stage": min(3, i // 4),
                  "VCI": 50.0, "SMI": 0.4, "DAS": 10 * i}
                 for i, (m, d) in enumerate([(6, 1), (6, 11), (6, 21), (7, 1), (7, 11), (7, 21),
                                             (8, 1), (8, 11), (8, 21), (9, 1), (9, 11), (9, 21)])]
        adv = [{"date": "2025-09-10", "advisory": 1, "deficit": 18.0, "ETc": 30.0, "ET0": 28.0,
                "rain": 15.0, "Kc": 1.1, "percolation": 0.0},
               {"date": "2025-11-16", "advisory": 2, "deficit": 33.0, "ETc": 34.0, "ET0": 30.0,
                "rain": 0.0, "Kc": 1.15, "percolation": 0.0}]
        return {"static": {"class": 2, "sowing_t": 30, "active": 1, "ndvi_max": 0.82,
                           "ndvi_amp": 0.55},
                "composites": comps, "advisory": adv, "season_start": "2025-06-01"}


@pytest.fixture()
def app():
    roi_info = {"geo": {"type": "Polygon", "coordinates": [[[77.6, 29.1], [77.7, 29.1],
                                                            [77.7, 29.2], [77.6, 29.1]]]},
                "area": 4e8, "c": [77.65, 29.15]}
    with mock.patch("pamrofi.earthengine.init"), \
         mock.patch("pamrofi.pipeline.Pipeline", FakePipeline), \
         mock.patch("ee.Dictionary") as D:
        D.return_value.getInfo.return_value = roi_info
        at = AppTest.from_file(APP, default_timeout=60)
        yield at


def test_app_runs_all_tabs(app):
    app.run()
    assert not app.exception, app.exception
    assert any("UNVALIDATED" in w.value for w in app.warning)
    assert len(app.tabs) == 6


def test_app_field_inspector_and_validation(app):
    app.run()
    app.session_state["field"] = (29.117, 77.685)
    app.run()
    assert not app.exception, app.exception
    # advisory message for the last period (default) is rendered
    html = " ".join(m.value for m in app.markdown)
    assert "Irrigate about 33 mm immediately" in html
    app.button[0].click().run()  # "Run validation"
    assert not app.exception, app.exception
    assert any(m.label.startswith("Overall accuracy") for m in app.metric)


def test_app_hindi(app):
    app.run()
    app.session_state["field"] = (29.117, 77.685)
    lang = [r for r in app.radio if r.label.startswith("Farmer")][0]
    lang.set_value("हिन्दी").run()
    assert not app.exception, app.exception
    html = " ".join(m.value for m in app.markdown)
    assert "मिमी" in html
