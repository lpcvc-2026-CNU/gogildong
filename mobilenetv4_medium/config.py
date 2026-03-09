"""
YAML 기반 설정 로더.
config.yaml 을 읽어 중첩된 dict 를 속성 접근(dot notation) 가능한 객체로 반환합니다.

사용 예:
    from utils.config import load_config
    cfg = load_config("config.yaml")
    print(cfg.model.embed_dim)       # 512
    print(cfg.training.stage1.lr)    # 1e-4
"""

from __future__ import annotations

import yaml
from pathlib import Path
from typing import Any, Union


class ConfigNode:
    """
    중첩 dict 를 속성 접근으로 읽을 수 있게 감싸는 클래스.
    cfg["key"] 와 cfg.key 모두 지원합니다.
    """

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
        """ConfigNode → dict 로 재변환."""
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
    YAML 파일을 읽어 ConfigNode 객체로 반환합니다.

    Args:
        config_path: config.yaml 경로 (기본값: "config.yaml")

    Returns:
        ConfigNode: 속성 접근 가능한 설정 객체

    Raises:
        FileNotFoundError: config.yaml 이 존재하지 않을 때
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(
            f"설정 파일을 찾을 수 없습니다: {path.resolve()}\n"
            f"프로젝트 루트에 config.yaml 이 있는지 확인하세요."
        )

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if raw is None:
        raise ValueError(f"config.yaml 이 비어 있습니다: {path}")

    return ConfigNode(raw)


def save_config(cfg: ConfigNode, output_path: Union[str, Path]) -> None:
    """ConfigNode 를 YAML 파일로 저장합니다 (학습 재현용)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg.to_dict(), f, default_flow_style=False, allow_unicode=True)
