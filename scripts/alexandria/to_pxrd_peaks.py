from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import polars as pl
from tqdm import tqdm

from material_database.constants import ColumnNames
from material_database.iterate import DatabaseFileIterator
from material_database.pxrd.peak_calculation import (
    calculate_peaks_from_symmetrized_structure_entry,
    writeout_constant_metadata,
)
from material_database.types_ import SerializedPXRDPeaks


def main():
    source = Path(
        "/p/project1/solai/oestreicher1/repos/material_database/data/alexandria"
    )

    destination = source / "pxrd_peaks"
    destination.mkdir(exist_ok=True, parents=True)
    writeout_constant_metadata(destination)

    for file, df in tqdm(
        DatabaseFileIterator(source=source / "pymatgen"), desc="Processing files"
    ):
        if (destination / file.name).exists():
            print(f"Skipping {file.name}, since it was already processed.")
            continue

        df = df.collect()

        ids = df[ColumnNames.ID]
        symmetrized_structures = df[ColumnNames.SYMMETRIZED_STRUCTURE]

        with ProcessPoolExecutor() as executor:
            peaks: list[SerializedPXRDPeaks] = list(
                tqdm(
                    executor.map(
                        calculate_peaks_from_symmetrized_structure_entry,
                        symmetrized_structures,
                    ),
                    total=len(symmetrized_structures),
                    desc=f"Calculating PXRD peaks for {file.name}",
                )
            )

        data = pl.DataFrame({ColumnNames.ID: ids}).with_columns(
            pl.DataFrame(
                peaks,
                schema=SerializedPXRDPeaks.get_schema(),
            )
        )
        data.write_parquet(destination / file.name)
        print(f"Wrote {destination / file.name}")


if __name__ == "__main__":
    main()
