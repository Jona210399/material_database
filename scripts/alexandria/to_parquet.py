from pathlib import Path

import polars as pl
from tqdm import tqdm

from material_database.alexandria.reading import read_raw_alexandria_json_bz2
from material_database.alexandria.sanitize import (
    sanitize_alexandria_entry,
)


def alexandria_raw_to_parquet(
    alexandria_raw_dir: Path,
    alexandria_raw_dir_parquet: Path,
    id_column: str,
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
                ).alias(id_column)
            ]
        )
        entries = entries.select(
            [id_column] + [col for col in entries.columns if col != id_column]
        )
        entries.write_parquet(
            alexandria_raw_dir_parquet / file.name.replace(".json.bz2", ".parquet")
        )
        start_idx += entries.height


def main():
    ALEXANDRIA_RAW_DIR = Path.cwd() / "data" / "alexandria" / "raw" / "json_bz2"
    ALEXANDRIA_RAW_DIR_PARQUET = Path.cwd() / "data" / "alexandria" / "raw" / "parquet"
    ID_COLUMN = "id"
    alexandria_raw_to_parquet(ALEXANDRIA_RAW_DIR, ALEXANDRIA_RAW_DIR_PARQUET, ID_COLUMN)


if __name__ == "__main__":
    main()

    """df = pl.scan_parquet(
        Path.cwd() / "data" / "alexandria" / "parquet_sanitized",
        cast_options=pl.ScanCastOptions(
            missing_struct_fields="insert",
            extra_struct_fields="ignore",
        ),
    )
    print(df)"""

    def duckdb_test():
        import duckdb

        db_path = Path.cwd() / "data" / "alexandria" / "alexandria.db"

        con = duckdb.connect(db_path)

        parquet_path = db_path.parent / "parquet"

        query = f"SELECT * FROM read_parquet('{parquet_path}/alexandria_*.parquet')"

        arrow = con.execute(query).arrow()

        df = con.execute(f"""
        SELECT entries
        FROM read_parquet('{parquet_path}/alexandria_*.parquet') 
        LIMIT 1
    """).fetchdf()

        print(df["entries"][0])
