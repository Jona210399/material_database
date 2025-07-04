import bz2
from pathlib import Path

import pandas as pd
import polars as pl


def read_raw_alexandria_json_bz2(filepath: Path) -> pl.Series:
    with bz2.open(filepath, mode="rb") as f:
        try:
            df = pl.read_json(f)
            if "entries" not in df.columns:
                raise ValueError(
                    f"Expected 'entries' column in {filepath}, found: {df.columns}"
                )

            return df.explode("entries").to_series()
        except pl.exceptions.ComputeError:
            f.seek(0)
            df = pd.read_json(f)
            return pl.from_pandas(df).to_series()


def remove_none_entries_from_entry_composition(alexandria_entry: dict) -> dict:
    composition: dict = alexandria_entry.get("composition", dict())
    for k, v in list(composition.items()):
        if v is None:
            composition.pop(k)
    alexandria_entry["composition"] = composition
    return alexandria_entry


def remove_empty_fields(d: dict) -> dict:
    if not isinstance(d, dict):
        return d

    new_dict = {}
    for k, v in d.items():
        if isinstance(v, dict):
            cleaned = remove_empty_fields(v)
            if cleaned:
                new_dict[k] = cleaned
        else:
            new_dict[k] = v
    return new_dict


def sanitize_alexandria_entry(entry: dict) -> dict:
    entry = remove_none_entries_from_entry_composition(entry)
    entry = remove_empty_fields(entry)
    return entry
