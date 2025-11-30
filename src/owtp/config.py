import yaml
from pathlib import Path
from typing import Union

def load_yaml_config(file_path: Union[str, Path] = 'config/config.yaml') -> dict:
    with open(file_path, 'r') as f:
        config = yaml.safe_load(f)
    return config