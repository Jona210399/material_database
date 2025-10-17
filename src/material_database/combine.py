from functools import reduce
from pathlib import Path
from typing import List

import polars as pl
from tqdm import tqdm


def combine_dataset(dataset_path: str | Path) -> None:
    dataset_path = Path(dataset_path)

    features_to_parquets = {}
    for path in dataset_path.iterdir():
        if path.name == "raw" or path.name == "combined":
            continue

        if not path.is_dir():
            continue

        print("Found feature folder:", path.name)

        feature_files = sorted(list(path.glob("*.parquet")))
        features_to_parquets[path.name] = feature_files

    destination = dataset_path / "combined"
    destination.mkdir(parents=True, exist_ok=True)

    for files in tqdm(zip(*features_to_parquets.values())):
        if (destination / files[0].name).exists():
            print("Skipping existing:", files[0].name)
            continue

        print("Merging files:", [f.name for f in files])
        files: List[Path]
        dfs = [pl.read_parquet(f) for f in files]
        print("Combining...")

        combined: pl.DataFrame = reduce(
            lambda left, right: left.join(right, on="id", how="full", coalesce=True),
            dfs,
        )

        # Casting is required to enable pyarrow to read the lists back correctly even if some of them are None.
        # Generally if there are None values present but they are typed as pl.Array, pyarrow will fail to read the file.
        combined = combined.with_columns(
            pl.col("intensities").cast(pl.List(pl.Float32)),
            pl.col("edge_index").cast(pl.List(pl.List(pl.Int32))),
        )
        print("Writing...")

        print(combined)
        combined.write_parquet(destination / files[0].name)
        print("Done.")


if __name__ == "__main__":
    dataset_path = (
        "/p/project1/solai/oestreicher1/repos/material_database/data/alexandria/"
    )
    combine_dataset(dataset_path)
