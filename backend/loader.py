from __future__ import annotations
"""Loads + validates SOP YAML. Fails loudly at startup, naming the file.
SopRegistry hot-reloads when files change, so a new SOP needs no restart or code edit;
a bad edit at runtime keeps the last good policy set and exposes last_error."""
import os
from pathlib import Path
import yaml
from pydantic import ValidationError
from .config import Config
from .models import SOP

DEFAULT_SOP_DIR = Path(__file__).parent / "sops"


class SopLoadError(Exception):
    pass


def load_sops(sop_dir: Path, cfg: Config) -> list[SOP]:
    files = sorted(Path(sop_dir).glob("*.y*ml"))
    if not files:
        raise SopLoadError(f"No SOP files found in {sop_dir}")
    sops: list[SOP] = []
    seen: dict[str, str] = {}
    tax = cfg.taxonomy
    for f in files:
        try:
            raw = yaml.safe_load(f.read_text())
        except yaml.YAMLError as e:
            raise SopLoadError(f"{f.name}: invalid YAML: {e}")
        if not isinstance(raw, list):
            raise SopLoadError(f"{f.name}: top level must be a list of SOPs")
        for i, item in enumerate(raw):
            try:
                sop = SOP.model_validate(item)
            except ValidationError as e:
                raise SopLoadError(f"{f.name} [entry {i}]: {e}")
            if sop.id in seen:
                raise SopLoadError(f"{f.name}: duplicate id {sop.id} (also in {seen[sop.id]})")
            seen[sop.id] = f.name
            problems = []
            if sop.conditions:
                bad = sop.conditions.referenced_fields() - cfg.sop_fields
                if bad:
                    problems.append(f"unknown weather fields {sorted(bad)} (allowed: weather_fields.yaml)")
            ap = sop.applies_to
            groups = {g for a in tax["activities"].values() for g in a["groups"]}
            if set(ap.activities) - set(tax["activities"]):
                problems.append(f"unknown activities {sorted(set(ap.activities) - set(tax['activities']))}")
            if set(ap.activity_groups) - groups:
                problems.append(f"unknown activity_groups {sorted(set(ap.activity_groups) - groups)}")
            if set(ap.audiences) - set(tax["audiences"]):
                problems.append(f"unknown audiences {sorted(set(ap.audiences) - set(tax['audiences']))}")
            if problems:
                raise SopLoadError(f"{f.name}: {sop.id}: " + "; ".join(problems))
            sops.append(sop)
    return sops


class SopRegistry:
    def __init__(self, cfg: Config, sop_dir: Path | None = None):
        self.cfg = cfg
        self.dir = Path(sop_dir or os.getenv("SOP_DIR") or DEFAULT_SOP_DIR)
        self.last_error: str | None = None
        self._sops = load_sops(self.dir, cfg)          # raises at startup
        self._sig = self._signature()

    def _signature(self):
        return tuple((p.name, p.stat().st_mtime_ns) for p in sorted(self.dir.glob("*.y*ml")))

    def get(self) -> list[SOP]:
        sig = self._signature()
        if sig != self._sig:
            self._sig = sig
            try:
                self._sops = load_sops(self.dir, self.cfg)
                self.last_error = None
            except SopLoadError as e:
                self.last_error = str(e)           # keep last good set
        return self._sops
