import math

import pytest

from pamrofi import agro
from pamrofi.classify import confusion_metrics
from pamrofi.config import Config, ConfigError


@pytest.fixture(scope="module")
def season():
    return Config.load().season("kharif_2025")


# ------------------------------------------------------------- Kc curve
def test_kc_paddy_fao56_shape(season):
    paddy = season.cls(2)               # ini 1.05, mid 1.20, end 0.75; 20/30/50/25
    assert agro.kc_at(-1, paddy) == 0.0
    assert agro.kc_at(0, paddy) == pytest.approx(1.05)
    assert agro.kc_at(20, paddy) == pytest.approx(1.05)
    assert agro.kc_at(35, paddy) == pytest.approx(1.05 + 15 * (0.15 / 30))   # mid-development
    assert agro.kc_at(50, paddy) == pytest.approx(1.20)
    assert agro.kc_at(100, paddy) == pytest.approx(1.20)
    assert agro.kc_at(125, paddy) == pytest.approx(0.75)                    # end of season
    assert agro.kc_at(126, paddy) == 0.0                                     # harvested


def test_kc_non_agri_is_zero(season):
    assert agro.kc_at(50, season.cls(0)) == 0.0


def test_kc_continuous(season):
    for c in season.classes:
        if not c.irrigable:
            continue
        for b in agro.stage_boundaries(c)[:3]:
            assert agro.kc_at(b, c) == pytest.approx(agro.kc_at(b + 1e-6, c), abs=1e-4)


def test_stage_tokens(season):
    paddy = season.cls(2)
    assert agro.stage_at(-5, paddy) == agro.STAGE_OUT_OF_SEASON
    assert agro.stage_at(10, paddy) == agro.STAGE_INITIAL
    assert agro.stage_at(30, paddy) == agro.STAGE_CRITICAL
    assert agro.stage_at(99, paddy) == agro.STAGE_CRITICAL
    assert agro.stage_at(110, paddy) == agro.STAGE_LATE
    assert agro.stage_at(200, paddy) == agro.STAGE_OUT_OF_SEASON


# ------------------------------------------------------------------ ET0
def test_ra_matches_fao56_example_8():
    # FAO-56 Example 8: 20°S on 3 September (DOY 246) → Ra = 32.2 MJ m-2 day-1
    assert agro.extraterrestrial_radiation(-20, 246) == pytest.approx(32.2, abs=0.1)


def test_hargreaves_reasonable_for_meerut_september():
    et0 = agro.hargreaves_et0(33, 25, 29.1, 253)
    assert 3.5 < et0 < 6.0          # typical Western UP monsoon-end ET0, mm/day


# -------------------------------------------------------- water balance
def test_deficit_and_advisory():
    # ET0 40 mm/8d, Kc 1.2 → ETc 48; rain 20 → eff 16; deficit 32 → immediate
    d = agro.water_deficit(40, 1.2, 20, 0.8)
    assert d == pytest.approx(32)
    assert agro.advisory_level(d) == 2
    assert agro.advisory_level(30) == 1
    assert agro.advisory_level(10) == 1
    assert agro.advisory_level(9.99) == 0
    assert agro.water_deficit(10, 1.0, 100) == 0          # floored at zero
    assert agro.water_deficit(40, 1.2, 0, percolation_mm=24) == pytest.approx(72)


def test_messages_both_languages():
    adv = agro.FieldAdvice("Paddy (rice)", 2, 24.4, 1, "2025-09-10", "2025-09-17")
    en = adv.message("en")
    hi = adv.message("hi")
    assert "24 mm" in en and "3–4 days" in en
    assert "24 मिमी" in hi


# ------------------------------------------------------- stress matrix
def test_stress_matrix_is_phase_aware():
    S = agro
    assert S.stress_level(S.STAGE_OUT_OF_SEASON, 10, 0.1) is None
    # initial stage: low VCI ignored, flooded paddy never stressed
    assert S.stress_level(S.STAGE_INITIAL, vci=5, smi=0.8) == 0
    assert S.stress_level(S.STAGE_INITIAL, vci=90, smi=0.2) == 2
    assert S.stress_level(S.STAGE_INITIAL, vci=90, smi=0.2, flooded=True) == 0
    # critical stage: full matrix
    assert S.stress_level(S.STAGE_CRITICAL, vci=30, smi=0.9) == 2
    assert S.stress_level(S.STAGE_CRITICAL, vci=55, smi=0.45) == 1
    assert S.stress_level(S.STAGE_CRITICAL, vci=80, smi=0.8) == 0
    # late stage: senescence alone is not stress
    assert S.stress_level(S.STAGE_LATE, vci=20, smi=0.8) == 0
    assert S.stress_level(S.STAGE_LATE, vci=20, smi=0.2) == 1


# --------------------------------------------------------------- metrics
def test_confusion_metrics():
    m = confusion_metrics([[50, 0], [10, 40]], ["a", "b"])
    assert m["overall_accuracy"] == pytest.approx(0.9)
    # pe = (50*60 + 50*40) / 100^2 = 0.5 → kappa 0.8
    assert m["kappa"] == pytest.approx(0.8)
    assert m["per_class"]["a"]["producers_accuracy"] == pytest.approx(1.0)
    assert m["per_class"]["a"]["users_accuracy"] == pytest.approx(50 / 60)
    assert math.isclose(m["per_class"]["b"]["f1"], 2 * 0.8 * 1.0 / 1.8)


# ---------------------------------------------------------------- config
def test_config_loads_all_seasons():
    cfg = Config.load()
    for key in cfg.season_keys:
        s = cfg.season(key)
        assert [c.id for c in s.classes] == list(range(len(s.classes)))


def test_config_rejects_bad_thresholds():
    raw = Config.load().raw
    raw["advisory"]["thresholds_mm"] = [30, 10]
    with pytest.raises(ConfigError):
        Config(raw)


def test_overrides_do_not_mutate_original():
    cfg = Config.load()
    new = cfg.with_overrides(classification={"labels": "labels/x.geojson"})
    assert new.section("classification")["labels"] == "labels/x.geojson"
    assert cfg.section("classification")["labels"] == "weak"
