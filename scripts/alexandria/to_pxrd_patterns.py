import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import polars as pl
from tqdm import tqdm

from material_database.constants import ColumnNames
from material_database.iterate import DatabaseFileIterator
from material_database.pxrd.peak_convolution import (
    convolve_serialized_peaks,
    writeout_constant_metadata,
)
from material_database.types_ import SerializedPXRDGaussianScherrerProfile


def seed_process():
    seed = os.getpid()
    np.random.seed(seed)


def main():
    source = Path(
        "/p/project1/solai/oestreicher1/repos/material_database/data/alexandria"
    )

    destination = source / "pxrd_gaussian_scherrer"
    destination.mkdir(exist_ok=True, parents=True)
    writeout_constant_metadata(destination)

    for file, df in tqdm(
        DatabaseFileIterator(source=source / "pxrd_peaks"), desc="Processing files"
    ):
        if (destination / file.name).exists():
            print(f"Skipping {file.name}, since it was already processed.")
            continue

        df = df.collect()

        ids = df[ColumnNames.ID]
        peak_two_thetas = df[ColumnNames.PEAK_TWO_THETAS]
        peak_intensities = df[ColumnNames.PEAK_INTENSITIES]

        with ProcessPoolExecutor(initializer=seed_process) as executor:
            patterns: list[SerializedPXRDGaussianScherrerProfile] = list(
                tqdm(
                    executor.map(
                        convolve_serialized_peaks,
                        peak_two_thetas,
                        peak_intensities,
                    ),
                    total=len(peak_two_thetas),
                    desc=f"Calculating PXRD Patterns for {file.name}",
                )
            )

        data = pl.DataFrame({ColumnNames.ID: ids}).with_columns(
            pl.DataFrame(
                patterns,
                schema=SerializedPXRDGaussianScherrerProfile.get_schema(
                    intensities_shape=(len(patterns[0]["intensities"]),)
                ),
            )
        )

        num_invalid = data.select(pl.col("intensities")).null_count().item()
        if num_invalid > 0:
            print(f"Found {num_invalid} invalid patterns in {file.name}")

        data = data.drop_nulls()
        data.write_parquet(destination / file.name)


if __name__ == "__main__":
    main()
