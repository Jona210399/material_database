from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import polars as pl
from pymatgen.entries.computed_entries import (
    ComputedStructureEntry,
)
from tqdm import tqdm

from material_database.alexandria.parse import alexandria_entry_to_pymatgen
from material_database.constants import ColumnNames
from material_database.iterate import DatabaseFileIterator
from material_database.serialization import (
    structure_to_serialized_symmetrized_structure_and_cif,
)


def process_entry(
    entry: ComputedStructureEntry,
) -> tuple[dict, str] | tuple[None, None]:
    return structure_to_serialized_symmetrized_structure_and_cif(
        structure=entry.structure
    )


def process_entries_concurrently(
    entries: list[ComputedStructureEntry],
) -> tuple[list, list]:
    with ProcessPoolExecutor() as executor:
        results = list(
            tqdm(
                executor.map(process_entry, entries),  # ordered
                total=len(entries),
                desc="Processing entries",
            )
        )
    symmetrized_structures, cifs = zip(*results)
    return list(symmetrized_structures), list(cifs)


def process_entries(
    entries: list[ComputedStructureEntry],
) -> tuple[list, list]:
    symmetrized_structures = []
    cifs = []

    for entry in tqdm(entries, desc="Processing entries"):
        symmetrized_structure, cif = process_entry(entry=entry)
        symmetrized_structures.append(symmetrized_structure)
        cifs.append(cif)

    return symmetrized_structures, cifs


def main():
    source = Path.cwd() / "data" / "alexandria" / "raw" / "parquet"
    pymatgen_destination = Path.cwd() / "data" / "alexandria" / "pymatgen"
    cif_destination = Path.cwd() / "data" / "alexandria" / "cif"

    for file, data in tqdm(
        DatabaseFileIterator(source=source), desc="Processing files"
    ):
        if (cif_destination / file.name).exists():
            print(f"Skipping {file.name}, already processed.")
            continue

        data = data.collect()

        data = data.with_columns(
            pl.col("entries").map_elements(
                alexandria_entry_to_pymatgen, return_dtype=pl.Object
            ),
        ).drop_nulls()

        symmetrized_structures, cifs = process_entries_concurrently(
            data["entries"].to_list()
        )

        data.with_columns(
            pl.Series(
                name=ColumnNames.SYMMETRIZED_STRUCTURE,
                values=symmetrized_structures,
            ),
        ).drop("entries").drop_nulls().write_parquet(pymatgen_destination / file.name)

        data.with_columns(
            pl.Series(
                name=ColumnNames.CIF,
                values=cifs,
            ),
        ).drop("entries").drop_nulls().write_parquet(cif_destination / file.name)


if __name__ == "__main__":
    main()
