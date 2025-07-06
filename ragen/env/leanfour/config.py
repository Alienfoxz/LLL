from typing import Optional, List, Dict
from dataclasses import dataclass, field

@dataclass
class LeanfourEnvConfig:
    """Configuration for Lean4 environment"""
    # Map config
    cache_dir:str = field(default="./data")
    split: str = field(default="train")
    lean4api_url: str = field(default="http://10.82.1.240:5000/api/lean/evaluate")
    dataset_path: str = field(default="/home/panjieyu/ProjectRed/AAAI/dataset_processing/selected")

