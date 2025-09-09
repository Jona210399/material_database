from pathlib import Path

import polars as pl
from tqdm import tqdm

from material_database.alexandria.reading import read_raw_alexandria_json_bz2
from material_database.alexandria.sanitize import (
    sanitize_alexandria_entry,
)
from material_database.constants import ColumnNames


def alexandria_raw_to_parquet(
    alexandria_raw_dir: Path,
    alexandria_raw_dir_parquet: Path,
):
    if not alexandria_raw_dir.exists():
        raise FileNotFoundError(f"Raw directory does not exist: {alexandria_raw_dir}")

    if not alexandria_raw_dir_parquet.exists():
        alexandria_raw_dir_parquet.mkdir(parents=True, exist_ok=True)

    start_idx = 0
    for file in tqdm(sorted(alexandria_raw_dir.glob("*.json.bz2"))):
        entries = read_raw_alexandria_json_bz2(file)
        entries = (
            entries.drop_nulls()
            .map_elements(sanitize_alexandria_entry, return_dtype=pl.Struct)
            .to_frame()
        )
        entries = entries.with_columns(
            [
                pl.format(
                    "alex-{}", pl.arange(start_idx, start_idx + entries.height)
                ).alias(ColumnNames.ID)
            ]
        )
        entries = entries.select(
            [ColumnNames.ID] + [col for col in entries.columns if col != ColumnNames.ID]
        )
        entries.write_parquet(
            alexandria_raw_dir_parquet / file.name.replace(".json.bz2", ".parquet")
        )
        start_idx += entries.height


def main():
    ALEXANDRIA_RAW_DIR = Path.cwd() / "data" / "alexandria" / "raw" / "json_bz2"
    ALEXANDRIA_RAW_DIR_PARQUET = Path.cwd() / "data" / "alexandria" / "raw" / "parquet"
    alexandria_raw_to_parquet(ALEXANDRIA_RAW_DIR, ALEXANDRIA_RAW_DIR_PARQUET)


if __name__ == "__main__":
    main()
