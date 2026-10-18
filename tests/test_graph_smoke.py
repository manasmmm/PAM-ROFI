"""Smoke test: build the whole Earth Engine computation graph with a mocked
``ee`` module. No network or credentials needed. It catches Python-level
mistakes (typos, wrong arguments, broken control flow) in every code path that
builds images; the real server-side behaviour is checked by running the app."""
from unittest import mock

import pytest

from pamrofi.config import Config


@pytest.fixture()
def fake_ee(monkeypatch):
    fake = mock.MagicMock(name="ee")
    import pamrofi.advisory as a
    import pamrofi.classify as c
    import pamrofi.data as d
    import pamrofi.features as f
    import pamrofi.phenology as p
    import pamrofi.pipeline as pl
    import pamrofi.stress as s
    for mod in (a, c, d, f, p, pl, s):
        monkeypatch.setattr(mod, "ee", fake)
    return fake


@pytest.mark.parametrize("season", ["kharif_2025", "rabi_2025_26"])
@pytest.mark.parametrize("labels", ["weak", "tests/fixtures/example_labels.geojson",
                                    "projects/x/assets/labels"])
@pytest.mark.parametrize("et0", ["hargreaves_era5", "modis_pet"])
def test_full_graph_builds(fake_ee, season, labels, et0):
    from pamrofi.pipeline import Pipeline, PipelineOptions
    cfg = Config.load().with_overrides(classification={"labels": labels},
                                       advisory={"et0_source": et0, "rain_source": "era5"})
    for roi in cfg.rois:
        pipe = Pipeline(cfg, PipelineOptions(season_key=season, roi_key=roi))
        pipe.stress("2025-09-10")
        pipe.advisory("2025-09-10")
        pipe.sowing_doy()
        pipe.cube.classification_stack()
        assert pipe.validated == (labels != "weak")


def test_kc_expression_is_valid_text(fake_ee):
    """The inlined Kc expression must only reference D and numbers."""
    from pamrofi.phenology import PhenologyModel
    from pamrofi.features import SeasonCube
    season = Config.load().season("kharif_2025")
    cube = SeasonCube(fake_ee.Geometry.Point([0, 0]), season.start, season.end)
    pm = PhenologyModel(season, cube, fake_ee.Image.constant(1))
    das_img = fake_ee.Image.constant.return_value.float.return_value.subtract.return_value.rename.return_value
    das_img.expression.reset_mock()
    pm.kc("2025-09-10")
    exprs = [call.args[0] for call in das_img.expression.call_args_list]
    assert len(exprs) == 3                       # one per irrigable class
    for e in exprs:
        tokens = e.replace("?", " ").replace(":", " ").replace("(", " ").replace(")", " ")
        for tok in tokens.split():
            assert tok in {"D", "<", "<=", "+", "-", "*"} or float(tok) is not None
