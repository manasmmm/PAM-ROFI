"""Configuration loading and validation (pure Python, no Earth Engine needed)."""
from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"
VALID_SOWING_METHODS = {"vh_min", "ndvi_sos", "fixed"}


class ConfigError(ValueError):
    """Raised when config.yaml is missing something the pipeline needs."""


@dataclass
class CropClass:
    id: int
    name: str
    color: str
    irrigable: bool = False
    sowing_method: str | None = None
    fixed_sowing: str | None = None
    sos_fraction: float = 0.35
    emergence_lag_days: int = 0
    fallow_amplitude: float | None = None
    percolation_mm_day: float = 0.0
    kc: dict[str, float] = field(default_factory=dict)
    stages: dict[str, int] = field(default_factory=dict)

    @property
    def season_length(self) -> int:
        return int(sum(self.stages.values())) if self.stages else 0


@dataclass
class Season:
    key: str
    label: str
    start: str
    end: str
    composite_days: int
    sowing_search: tuple[str, str]
    classes: list[CropClass]

    def cls(self, class_id: int) -> CropClass:
        for c in self.classes:
            if c.id == class_id:
                return c
        raise KeyError(class_id)

    @property
    def class_names(self) -> dict[int, str]:
        return {c.id: c.name for c in self.classes}

    @property
    def palette(self) -> list[str]:
        return [c.color for c in sorted(self.classes, key=lambda c: c.id)]


def _parse_date(s: str, where: str) -> date:
    try:
        return date.fromisoformat(str(s))
    except ValueError as exc:
        raise ConfigError(f"{where}: '{s}' is not an ISO date (YYYY-MM-DD)") from exc


def _build_class(raw: dict, where: str) -> CropClass:
    c = CropClass(**{k: v for k, v in raw.items() if k in CropClass.__dataclass_fields__})
    if c.irrigable:
        if c.sowing_method not in VALID_SOWING_METHODS:
            raise ConfigError(f"{where}: sowing_method must be one of {sorted(VALID_SOWING_METHODS)}")
        for key in ("ini", "mid", "end"):
            if key not in c.kc:
                raise ConfigError(f"{where}: kc.{key} missing")
        for key in ("ini", "dev", "mid", "late"):
            if key not in c.stages or int(c.stages[key]) <= 0:
                raise ConfigError(f"{where}: stages.{key} must be a positive number of days")
        if c.sowing_method == "fixed":
            if not c.fixed_sowing:
                raise ConfigError(f"{where}: fixed sowing needs fixed_sowing")
            _parse_date(c.fixed_sowing, where)
    return c


class Config:
    """Thin wrapper around the YAML dict with validated accessors."""

    def __init__(self, raw: dict[str, Any]):
        self.raw = raw
        self._validate()

    @classmethod
    def load(cls, path: str | os.PathLike | None = None) -> "Config":
        path = Path(path) if path else DEFAULT_CONFIG_PATH
        if not path.exists():
            raise ConfigError(f"config file not found: {path}")
        with open(path, encoding="utf-8") as fh:
            return cls(yaml.safe_load(fh))

    # ------------------------------------------------------------------ access
    @property
    def ee_project(self) -> str | None:
        return os.environ.get("PAMROFI_EE_PROJECT") or self.raw.get("ee_project") or None

    @property
    def rois(self) -> dict[str, dict]:
        return self.raw["rois"]

    def season(self, key: str | None = None) -> Season:
        key = key or self.raw["default_season"]
        s = self.raw["seasons"][key]
        return Season(
            key=key,
            label=s.get("label", key),
            start=s["start"],
            end=s["end"],
            composite_days=int(s.get("composite_days", 10)),
            sowing_search=tuple(s.get("sowing_search", [s["start"], s["end"]])),
            classes=[_build_class(c, f"seasons.{key}.classes[{i}]") for i, c in enumerate(s["classes"])],
        )

    @property
    def season_keys(self) -> list[str]:
        return list(self.raw["seasons"].keys())

    def section(self, name: str) -> dict:
        return copy.deepcopy(self.raw.get(name, {}))

    def with_overrides(self, **sections: dict) -> "Config":
        """Return a copy with some sections updated (used by the app sidebar)."""
        raw = copy.deepcopy(self.raw)
        for name, values in sections.items():
            raw.setdefault(name, {}).update(values)
        return Config(raw)

    # -------------------------------------------------------------- validation
    def _validate(self) -> None:
        for key in ("rois", "seasons", "default_season", "default_roi"):
            if key not in self.raw:
                raise ConfigError(f"config is missing '{key}'")
        if self.raw["default_roi"] not in self.raw["rois"]:
            raise ConfigError("default_roi is not one of rois")
        if self.raw["default_season"] not in self.raw["seasons"]:
            raise ConfigError("default_season is not one of seasons")
        for key in self.raw["seasons"]:
            season = self.season(key)
            start = _parse_date(season.start, f"seasons.{key}.start")
            end = _parse_date(season.end, f"seasons.{key}.end")
            if end <= start:
                raise ConfigError(f"seasons.{key}: end must be after start")
            ids = sorted(c.id for c in season.classes)
            if ids != list(range(len(ids))):
                raise ConfigError(f"seasons.{key}: class ids must be 0..N-1 without gaps")
        th = self.raw.get("advisory", {}).get("thresholds_mm", [10, 30])
        if len(th) != 2 or not th[0] < th[1]:
            raise ConfigError("advisory.thresholds_mm must be two increasing numbers")
