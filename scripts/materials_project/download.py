import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import polars as pl
from dotenv import dotenv_values
from mp_api.client import MPRester
from pymatgen.core.structure import Structure
from tqdm import tqdm

from material_database.constants import ColumnNames
from material_database.serialization import (
    structure_to_serialized_symmetrized_structure_and_cif,
)
from material_database.utils.url_download import write_time_stamp

MATERIALS_PROJECT_API_KEY = dotenv_values().get("MATERIALS_PROJECT_API_KEY")


def process_structure(structure: Structure) -> tuple[dict, str] | tuple[None, None]:
    try:
        return structure_to_serialized_symmetrized_structure_and_cif(structure)
    except ValueError:
        # Symmetry detection can fail for some structures
        return None, None


def ignore_pymatgen_warnings():
    warnings.filterwarnings("ignore", category=UserWarning, module="pymatgen")


def get_structures(
    num_structures: int | None = None,
) -> tuple[list[Structure], list[str]]:
    if num_structures:
        chunk_size = min(1000, num_structures)
        num_chunks = (num_structures + chunk_size - 1) // chunk_size
    else:
        # fetch all available structures
        chunk_size = 1000
        num_chunks = None

    with MPRester(MATERIALS_PROJECT_API_KEY) as mpr:
        materials = mpr.materials.search(
            num_chunks=num_chunks,
            chunk_size=chunk_size,
            fields=["material_id", "structure"],
        )

    structures = [material.structure for material in materials]
    ids = [material.material_id for material in materials]
    return structures, ids


def process_structures_concurrently(
    structures: list[Structure],
) -> tuple[list, list]:
    with ProcessPoolExecutor(
        max_workers=4, initializer=ignore_pymatgen_warnings
    ) as executor:
        results = list(
            tqdm(
                executor.map(process_structure, structures),  # ordered
                total=len(structures),
                desc="Serializing structures",
            )
        )
    serialized_structures, cifs = zip(*results)
    return list(serialized_structures), list(cifs)


def process_structures(structures: list[Structure]):
    serialized_structures = [None] * len(structures)
    cifs = [None] * len(structures)

    for i, structure in enumerate(tqdm(structures, desc="Serializing structures")):
        serialized_structure, cif = process_structure(structure)
        serialized_structures[i] = serialized_structure
        cifs[i] = cif

    return serialized_structures, cifs


def main():
    ignore_pymatgen_warnings()
    destination_folder = Path.cwd() / "data" / "materials_project"
    structures, ids = get_structures()

    serialized_structures, cifs = process_structures_concurrently(structures)

    structures = pl.DataFrame(
        {
            ColumnNames.ID: ids,
            ColumnNames.SYMMETRIZED_STRUCTURE: serialized_structures,
        }
    ).drop_nulls()

    cifs = pl.DataFrame(
        {
            ColumnNames.ID: ids,
            ColumnNames.CIF: cifs,
        }
    ).drop_nulls()

    structures.write_parquet(
        destination_folder / "pymatgen" / "materials_project_000.parquet", mkdir=True
    )

    cifs.write_parquet(
        destination_folder / "cifs" / "materials_project_000.parquet", mkdir=True
    )
    write_time_stamp(destination_folder=destination_folder)


if __name__ == "__main__":
    main()
