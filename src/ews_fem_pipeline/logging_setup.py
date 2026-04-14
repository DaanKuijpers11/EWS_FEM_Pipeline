import logging
import logging.config
import tomllib
from pathlib import Path


def setup_logging():
    config_path = Path(__file__).resolve().parent / "logging_config.toml"

    print("CONFIG PATH:", config_path)
    print("EXISTS:", config_path.exists())

    with open(config_path, "rb") as f:
        config = tomllib.load(f)

    logging.config.dictConfig(config)