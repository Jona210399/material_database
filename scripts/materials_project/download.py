from dotenv import dotenv_values
from mp_api.client import MPRester
from tqdm import tqdm

from material_database.serialization import (
    structure_to_serialized_symmetrized_structure_and_cif,
)

MATERIALS_PROJECT_API_KEY = dotenv_values().get("MATERIALS_PROJECT_API_KEY")

i = 0
with MPRester(MATERIALS_PROJECT_API_KEY) as mpr:
    materials = mpr.materials.search(crystal_system="Cubic", num_sites=(2, 2))

    for material in tqdm(materials, desc="Downloading structures"):
        i += 1
        # The 'structure' attribute contains the pymatgen Structure object
        structure = material.structure
        serialized_structure, cif = (
            structure_to_serialized_symmetrized_structure_and_cif(structure)
        )

        print(f"Downloaded structure for material ID: {material.material_id}")
        print(f"Serialized Structure: {serialized_structure}")
        print(f"CIF:\n{cif}\n")
        if i >= 10:
            break
