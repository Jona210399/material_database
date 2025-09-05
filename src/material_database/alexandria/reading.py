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
