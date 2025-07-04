from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import polars as pl
from pymatgen.entries.computed_entries import (
    ComputedStructureEntry,
)
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer, SpacegroupOperations
from pymatgen.symmetry.groups import SpaceGroup
from pymatgen.symmetry.structure import SymmetrizedStructure
from tqdm import tqdm

from material_database.alexandria.parse import alexandria_entry_to_pymatgen
from material_database.alexandria.sanitize import remove_empty_fields
from material_database.cif.writing import analyzer_to_cif


def symmetrized_structure_from_serialized(entry: dict) -> SymmetrizedStructure:
    spacegroup = SpaceGroup.from_int_number(entry["spacegroup"])
    entry["spacegroup"] = SpacegroupOperations(
        int_number=spacegroup.int_number,
        int_symbol=spacegroup.full_symbol,
        symmops=list(spacegroup.symmetry_ops),
    )
    return SymmetrizedStructure.from_dict(entry)


def process_entry(
    entry: ComputedStructureEntry,
) -> tuple[dict, str]:
    analyzer = SpacegroupAnalyzer(entry.structure)
    symmetrized_structure = analyzer.get_symmetrized_structure()
    symmetrized_structure = symmetrized_structure.as_dict()
    symmetrized_structure["equivalent_positions"] = symmetrized_structure[
        "equivalent_positions"
    ].tolist()
    symmetrized_structure = remove_empty_fields(symmetrized_structure)
    symmetrized_structure["spacegroup"] = analyzer.get_space_group_number()
    cif = analyzer_to_cif(analyzer=analyzer)

    return symmetrized_structure, cif


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

    for file in tqdm(sorted(source.glob("*.parquet"))):
        data = pl.read_parquet(file)
        data = data.with_columns(
            pl.col("entries").map_elements(
                alexandria_entry_to_pymatgen, return_dtype=pl.Object
            ),
        ).drop_nulls()

        symmetrized_structures, cifs = process_entries_concurrently(
            data["entries"].to_list()
        )

        symmetrized_structures = (
            pl.Series(
                name="symmetrized_structure",
                values=symmetrized_structures,
            ),
        )

        data.with_columns(symmetrized_structures).drop("entries").write_parquet(
            pymatgen_destination / file.name
        )

        data.with_columns(
            pl.Series(name="cif", values=cifs),
        ).drop("entries").write_parquet(cif_destination / file.name)


if __name__ == "__main__":
    main()
