# castle/__init__.py
from __future__ import annotations

# Re-export key building blocks
from .castle import CASTLE
from .config import load_config, save_config
from .helper import LearnableTimeScale, MultiTimeScale
from .decay_aware import DecayAware
from .time_embedding import TimeEmbeddingGRUEncoder
from .tabular import TabularEncoder
from .stream_cond import StreamCondBuilder
from .fusion import FusionAllStream
from .anchor import AnchorCreator
from .volatility_analyser import VolatilityAnalyzer, ResidualReliability
from .regime import RegimeAdapter
from .head import ConsensusTrustHead

__all__ = [
    "CASTLE", "load_config", "save_config",
    "LearnableTimeScale", "MultiTimeScale",
    "DecayAware", "TimeEmbeddingGRUEncoder", "TabularEncoder",
    "StreamCondBuilder", "FusionAllStream",
    "AnchorCreator", "VolatilityAnalyzer", "ResidualReliability",
    "RegimeAdapter", "ConsensusTrustHead",
    "load", "build_model",
    "__version__",
]

__version__ = "0.1.0"

def build_model(cfg: dict) -> CASTLE:
    """Build a CASTLE model from an in-memory config dict."""
    return CASTLE(cfg)

def load(config: str | dict, *, device: str | None = None) -> CASTLE:
    cfg = load_config(config) if not isinstance(config, dict) else dict(config)
    if device is not None:
        cfg["device"] = device
    return build_model(cfg)
