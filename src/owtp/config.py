import yaml
from pathlib import Path
from typing import Union

def load_yaml_config(file_path: Union[str, Path] = 'config/config.yaml') -> dict:
    if isinstance(file_path, str):
        file_path = Path(file_path)

    with open(file_path.resolve(), 'r') as f:
        config = yaml.safe_load(f)
    return config