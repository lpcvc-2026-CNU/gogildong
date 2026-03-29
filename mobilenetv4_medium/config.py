"""
YAML-based configuration utilities.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Union

import yaml


class ConfigNode:
    """Dictionary wrapper that supports attribute access."""

    def __init__(self, data: dict):
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, ConfigNode(value))
            else:
                setattr(self, key, value)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def to_dict(self) -> dict:
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, ConfigNode):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result

    def __repr__(self) -> str:
        return f"ConfigNode({self.to_dict()})"


def load_config(config_path: Union[str, Path] = "config.yaml") -> ConfigNode:
    """
    Load YAML and return a ConfigNode.

    Resolution order:
    1) provided path as-is (absolute, or relative to current working directory)
    2) when relative and not found, relative to this module directory
    """
    path = Path(config_path)
    candidates = [path]

    if not path.is_absolute():
        candidates.append(Path(__file__).resolve().parent / path)

    resolved: Path | None = None
    for candidate in candidates:
        if candidate.exists():
            resolved = candidate
            break

    if resolved is None:
        searched = "\n".join(f"- {c.resolve()}" for c in candidates)
        raise FileNotFoundError(
            "설정 파일을 찾을 수 없습니다.\n"
            f"검색한 경로:\n{searched}"
        )

    with open(resolved, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if raw is None:
        raise ValueError(f"config.yaml 파일이 비어 있습니다: {resolved}")

    cfg = ConfigNode(raw)

    # Backward-compatible default: if not set, use full train set.
    if hasattr(cfg, "data") and not hasattr(cfg.data, "train_subset_ratio"):
        setattr(cfg.data, "train_subset_ratio", 1.0)

    return cfg


def save_config(cfg: ConfigNode, output_path: Union[str, Path]) -> None:
    """Save ConfigNode to YAML."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg.to_dict(), f, default_flow_style=False, allow_unicode=True)
