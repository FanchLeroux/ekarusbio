from dataclasses import dataclass

import pathlib


@dataclass
class Config:
    root_dir: pathlib.Path = pathlib.Path(__file__).parents[2]
    width_single_column: float = 3.46  # [Inches]
    width_double_column: float = 7.09  # [Inches]
