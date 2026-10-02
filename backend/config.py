from __future__ import annotations
"""Loads the data-only config (weather fields, activity taxonomy)."""
from pathlib import Path
import yaml

CONFIG_DIR = Path(__file__).parent / "config"


class Config:
    def __init__(self, fields: dict, taxonomy: dict):
        self.fields, self.taxonomy = fields, taxonomy

    @property
    def sop_fields(self) -> set[str]:
        return set(self.fields["hourly"]) | {f"daily_{d}" for d in self.fields["daily"]} | {"local_hour"}

    def groups_for(self, activity: str) -> set[str]:
        return set(self.taxonomy["activities"].get(activity, {}).get("groups", []))


def load_config(config_dir: Path | None = None) -> Config:
    d = Path(config_dir) if config_dir else CONFIG_DIR
    read = lambda n: yaml.safe_load((d / n).read_text())
    return Config(read("weather_fields.yaml"), read("taxonomy.yaml"))
