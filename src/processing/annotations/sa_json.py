#!/usr/bin/env python3
"""sa_json.py in src/argusdp/processing/annotations."""

from pathlib import Path
from typing import Union

from argusdp.processing.annotations.base_annotation import BaseAnnotation


class SAJson(BaseAnnotation):
    """Class for handling Supervisely Annotation JSON files."""

    def __init__(self, filepath: Union[str, Path]):
        """Initialize SAJson."""
        super().__init__(filepath)
        self.height = self.data["metadata"]["height"]
        self.width = self.data["metadata"]["width"]
        self.name = self.data["metadata"]["name"]
        self.custom_metadata = self.data["metadata"]["custom_metadata"]
        self.num_instances = self._enumerate_instances(self.data.get("instances"))
