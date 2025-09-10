#!/usr/bin/env python3
"""base_annotation.py in src/argusdp/processing/annotations."""
import pprint
from pathlib import Path
from typing import Union

import pandas as pd

from argusdp.fileio.text.readers import json_loader
from argusdp.fileio.text.readers import yaml_loader
from argusdp.fileio.text.writers import json_writer
from argusdp.processing.annotations.mask import rle2mask
from argusdp.processing.image.imutils import image_show


class BaseAnnotation:
    """Base class for annotations."""

    def __init__(self, filepath: Union[str, Path]):
        """Initialize BaseAnnotation class."""
        self._filepath = Path(filepath).resolve()
        self.data = self._load_data()
        self.metadata = self.data.get("metadata", {})
        self.height = self.metadata.get("height", 0)
        self.width = self.metadata.get("width", 0)
        self.custom_metadata = self.data.get("custom_metadata", {})
        self.instances = self.data.get("instances", [])
        self.num_instances = self._enumerate_instances()
        self.tags = self.metadata.get("tags", [])
        self.rle_mask = self.custom_metadata.get("rle_mask", None)

    def __repr__(self) -> str:
        """An unambiguous string representation of the class instance."""
        return f"{self.__class__.__name__}({self.filepath})"

    def __str__(self) -> str:
        """An easy-to-read string representation of the class."""
        metadata_print = self.custom_metadata.copy()
        # remove large fields
        for k, v in metadata_print.items():
            if isinstance(v, str) and len(v) > 80:
                metadata_print[k] = f"{v[:80]}..."

        return (
            f"{self.__class__.__name__}\n\tFile: {self._filepath.name}\n"
            f"\tPath: {self._filepath.parent}\n"
            f"\tImage shape: {self.height} x {self.width}\n"
            f"\tNumber of instances: {len(self)}\n"
            f"\tCustom metadata ({len(self.custom_metadata)} fields):" + " {\n"
            f"{pprint.pformat(metadata_print, indent=8, width=80)[1:-1]}\n" + "\t}"
        )

    def __len__(self) -> int:
        """Return the number of instances in the annotation."""
        return len(self.instances)

    def _load_data(self) -> dict:
        """Load annotation data."""
        if self._filepath.suffix == ".json":
            return json_loader(self._filepath)
        elif self._filepath.suffix == ".yaml":
            return yaml_loader(self._filepath)
        else:
            raise ValueError(f"Invalid file type: {self._filepath.suffix}")

    def _enumerate_instances(self) -> dict:
        """Return an enumerated list of instances."""
        output_dict = {"all": len(self.instances)}
        if not self.instances:  # or filter_by is None:
            return output_dict

        # get classId, className and type for each instance
        # create DataFrame with columns: classId, className, type
        class_id = []
        class_name = []
        annotation_type = []
        for elem in self.instances:
            class_id.append(elem.get("classId", None))
            class_name.append(elem.get("className", None))
            annotation_type.append(elem.get("type", None))

        # create DataFrame of classId, className, type
        # group by classId, className, type and count
        # return as dictionary of counts
        self.df = pd.DataFrame(
            {"class_id": class_id, "class_name": class_name, "type": annotation_type}
        )
        # drop columns with all None values
        self.df = self.df.dropna(axis=1, how="all")

        # group by class_id, class_name, type and count
        df_grouped = self.df.groupby(list(self.df.columns)).size().reset_index()
        df_grouped.columns = list(self.df.columns) + ["count"]
        for _index, row in df_grouped.iterrows():
            output_dict[f"{row['class_name']}_{row['type']}"] = row["count"]

        return output_dict

    @property
    def filepath(self) -> str:
        """Return the annotation file path as string."""
        return self._filepath.as_posix()

    def to_dict(self) -> dict:
        """Convert annotation to dictionary."""
        return self.data

    def to_json(self, filepath: Union[str, Path]) -> None:
        """Save annotation to JSON file."""
        return json_writer(self.data, filepath)

    def to_df(self) -> None:
        """Convert annotation to a pandas DataFrame."""
        raise NotImplementedError("Subclasses must implement this method.")

    def show_mask(self, field: str = "rle_mask") -> None:
        """Display the binary foreground mask."""
        if field not in self.custom_metadata:
            raise ValueError(f"{field} not found in custom_metadata")

        rle_mask = self.custom_metadata[field]
        output_shape = (self.height, self.width)
        self.mask = rle2mask(rle_mask, output_shape)
        image_show(self.mask)
