from pydantic.dataclasses import dataclass


@dataclass
class LauncherConfig:
    kind: str


@dataclass
class ParslConfig:
    args: dict[str, int]
