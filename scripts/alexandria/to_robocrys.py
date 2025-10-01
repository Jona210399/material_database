from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import polars as pl
from tqdm import tqdm

from material_database.constants import ColumnNames
from material_database.iterate import DatabaseFileIterator
from material_database.to_robocrys import generate_text, mute_warnings


def main():
    source = Path(
        "/p/project1/solai/oestreicher1/repos/material_database/data/alexandria"
    )

    destination = source / "robocrys"
    destination.mkdir(exist_ok=True, parents=True)

    files_to_process = [15]

    print(f"Processing files: {files_to_process}")

    for i, (file, df) in enumerate(
        tqdm(DatabaseFileIterator(source=source / "pymatgen"), desc="Processing files")
    ):
        if i not in files_to_process:
            print(f"Skipping file {file.name}, since it's not in the list.")
            continue

        if (destination / file.name).exists():
            print(f"Skipping {file.name}, since it was already processed.")
            continue

        df = df.collect()

        ids = df[ColumnNames.ID]
        symmetrized_structures = df[ColumnNames.SYMMETRIZED_STRUCTURE]

        with ProcessPoolExecutor(initializer=mute_warnings) as executor:
            descriptions: list[str] = list(
                tqdm(
                    executor.map(
                        generate_text,
                        symmetrized_structures,
                    ),
                    total=len(symmetrized_structures),
                    desc=f"Generating robocrys descriptions for {file.name}",
                )
            )

        data = pl.DataFrame(
            {ColumnNames.ID: ids, ColumnNames.ROBOCRYS: descriptions}
        ).drop_nulls()

        data.write_parquet(destination / file.name)


if __name__ == "__main__":
    main()
