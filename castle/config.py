# castle/config.py
from __future__ import annotations
from pathlib import Path
import json

__all__ = ["load_config", "save_config"]

def _load_yaml(path: Path) -> dict:
    try:
        import yaml  # PyYAML
    except Exception as e:
        raise RuntimeError(
            "YAML config requested but PyYAML is not installed. "
            "Install with `pip install pyyaml` or provide a JSON file."
        ) from e
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def load_config(path_or_dict) -> dict:
    """
    Load a config from YAML/JSON file or pass-through a dict.
    - Accepts str/Path to *.yaml|*.yml|*.json
    - Returns a plain dict (no validation here; wiring has sensible defaults)
    """
    if isinstance(path_or_dict, dict):
        return dict(path_or_dict)
    p = Path(path_or_dict)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    suf = p.suffix.lower()
    if suf in (".yaml", ".yml"):
        return _load_yaml(p)
    if suf == ".json":
        return _load_json(p)
    raise ValueError(f"Unsupported config format: {p.suffix} (expected .yaml/.yml/.json)")

def save_config(cfg: dict, path: str | Path):
    """Save dict to YAML/JSON based on file suffix."""
    p = Path(path)
    suf = p.suffix.lower()
    if suf in (".yaml", ".yml"):
        try:
            import yaml
        except Exception as e:
            raise RuntimeError("Saving YAML requires `pyyaml` installed.") from e
        with p.open("w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
    elif suf == ".json":
        with p.open("w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
    else:
        raise ValueError(f"Unsupported config format for saving: {p.suffix}")
