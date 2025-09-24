from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import polars as pl
from tqdm import tqdm

from material_database.cif.parsing.file import CifFile
from material_database.cif.parsing.parser import ParsingSettings, parse_cif_file
from material_database.constants import ColumnNames
from material_database.serialization import serialize_symmetrized_structure


def cif_to_serialized_symmetrized_structure(cif: str) -> dict | None:
    cif = CifFile.from_str(cif)

    structures = parse_cif_file(
        cif_file=cif, parsing_settings=ParsingSettings(log_to_console=False)
    )

    if len(structures) == 0:
        # No structures found in CIF
        return None

    if len(structures) > 1:
        # More than one structure found in CIF, not supported
        print("More than one structure found in CIF, not supported.")
        return None

    structure = structures[0]

    if structure is None:
        # Parsing failed
        print("Parsing failed.")
        return None

    return serialize_symmetrized_structure(structure)


def process_entries(cifs: list[str]) -> list[dict]:
    symmetrized_structures = []

    for cif in tqdm(cifs, desc="Processing entries"):
        symmetrized_structure = cif_to_serialized_symmetrized_structure(cif)
        symmetrized_structures.append(symmetrized_structure)

    return symmetrized_structures


def process_entries_concurrently(cifs: list[str]) -> list[dict]:
    with ProcessPoolExecutor() as executor:
        symmetrized_structures = list(
            tqdm(
                executor.map(cif_to_serialized_symmetrized_structure, cifs),  # ordered
                total=len(cifs),
                desc="Processing entries",
            )
        )
    return symmetrized_structures


def main():
    source = Path.cwd() / "data" / "icsd"
    icsd = pl.read_parquet(source / "raw" / "icsd.parquet")

    ids = "icsd-" + icsd.select(pl.col("CollectionCode").cast(pl.Utf8)).to_series()

    cifs = pl.DataFrame(ids, schema=[ColumnNames.ID]).with_columns(
        icsd.select(pl.col(ColumnNames.CIF))
    )

    cifs.write_parquet(source / "cif" / "icsd_000.parquet", mkdir=True)

    symmetrized_structures = process_entries_concurrently(
        cifs.select(pl.col(ColumnNames.CIF)).to_series()
    )

    symmetrized_structures = cifs.select(pl.col(ColumnNames.ID)).with_columns(
        pl.Series(name=ColumnNames.SYMMETRIZED_STRUCTURE, values=symmetrized_structures)
    )

    symmetrized_structures = symmetrized_structures.drop_nulls()

    print(symmetrized_structures)

    symmetrized_structures.write_parquet(
        source / "pymatgen" / "icsd_000.parquet", mkdir=True
    )


if __name__ == "__main__":
    main()
