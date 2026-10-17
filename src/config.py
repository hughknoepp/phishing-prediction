from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"


def _load_yaml(relative_path: str) -> dict:
    with open(CONFIG_DIR / relative_path) as f:
        return yaml.safe_load(f)


def load_paths() -> dict[str, Path]:
    """Return configured data/model paths, resolved relative to the project root."""
    raw = _load_yaml("paths.yaml")
    return {key: PROJECT_ROOT / value for key, value in raw.items()}


def load_data_config() -> dict:
    """Return target column, dropped columns, and train/test/val split settings."""
    return _load_yaml("data_config.yaml")


def load_model_config(name: str) -> dict:
    """Return the model class name and hyperparameters for config/models/<name>.yaml."""
    return _load_yaml(f"models/{name}.yaml")
