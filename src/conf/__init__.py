#!/usr/bin/env python3
"""__init__.py in src/argusdp/conf."""

__all__ = ["STAINS", "DOMAINS", "MODALITIES", "VERSIONS"]

from pathlib import Path

import yaml


DOMAINS = Path(__file__).parent.joinpath("domains.yaml")
MODALITIES = Path(__file__).parent.joinpath("modalities.yaml")
STAINS = Path(__file__).parent.joinpath("stains.yaml")
TASKS = Path(__file__).parent.joinpath("tasks.yaml")
VERSIONS = Path(__file__).parent.joinpath("versions.yaml")

with open(DOMAINS) as f:
    DOMAINS = yaml.safe_load(f)

with open(MODALITIES) as f:
    MODALITIES = yaml.safe_load(f)

with open(STAINS) as f:
    STAINS = yaml.safe_load(f)

with open(TASKS) as f:
    TASKS = yaml.safe_load(f)

with open(VERSIONS) as f:
    VERSIONS = yaml.safe_load(f)
