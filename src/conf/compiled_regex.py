#!/usr/bin/env python3
"""compiled_regex.py in src/sngp_core/conf."""
import re


RE_UUID = re.compile(r"^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$")
RE_CCBY = re.compile(
    r"CC(-|\s)BY(-|\s)(NC|SA|ND)?(-|\s)?(NC|SA|ND)?(-|\s)?\d\.\d", re.IGNORECASE
)
RE_CC0 = re.compile(r"CC0((-|\s)\d\.\d)?", re.IGNORECASE)
RE_IMAGE_SHAPE = re.compile(r"^\[\d+,\s?\d+,?\s?\d?\]$")  # e.g., '[250, 250, 3]'
# check if image_mean_std is a valid List[list] formatted as str:
# regex to match a single floating point number
number_regex = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
# regex to match a list of such numbers within square brackets
float_list_regex = rf"\[\s*{number_regex}(?:\s*,\s*{number_regex})*\s*\]"
# regex to match the overall structure of nested lists
# nested_list_regex = rf'^{float_list_regex}\s*,\s*{float_list_regex}$'
RE_IMAGE_MEAN_STD = re.compile(
    rf"^\[?({float_list_regex})\s*,\s*({float_list_regex})\]?$"
)
RE_NAN = re.compile(r"nan|np.nan", re.IGNORECASE)
RE_FILENAME = re.compile(
    r"(?P<index>\d+)_"
    r"(?P<uuid>[a-f0-9]{8})_"
    r"(?P<patient_id>[a-zA-Z0-9\-]+)_"
    r"split-(?P<split>[a-z]+)_"
    r"(?P<label_name>[a-zA-Z0-9\-]+)\.(?P<ext>png|jpg|jpeg|gif|bmp|tiff|tif)"  # ome\.tiff|tif|ome\.tif
)
RE_RESERVED_CHARS = re.compile(r"[<>:\"/\\|?*]")
RE_VERSION = re.compile(r"(\d+\.\d+\.\d+)")
RE_LIST = re.compile(r"\[.*\]")
