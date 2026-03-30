from typing import Literal
import pathlib
from dataclasses import dataclass, Field, field, fields

import tyro


@dataclass
class OptimizerConfig:
    lr: float = 3e-4
    weight_decay: float = 1e-2


@dataclass
class Config:
    optimizer: OptimizerConfig
    seed: int = field(default=0)


def train(
    config: Config
):
    print("config: ", config)


if __name__ == '__main__':
    tyro.cli(train)

