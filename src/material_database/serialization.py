import numpy as np
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.symmetry.structure import SymmetrizedStructure

from material_database.cif.writing import analyzer_to_cif


def remove_empty_fields(obj):
    if isinstance(obj, dict):
        new_dict = {}
        for k, v in obj.items():
            cleaned = remove_empty_fields(v)
            # Only keep non-empty values
            if cleaned not in ({}, [], "", None):
                new_dict[k] = cleaned
        return new_dict

    elif isinstance(obj, list):
        new_list = [remove_empty_fields(v) for v in obj]
        # Drop empty items from lists
        return [v for v in new_list if v not in ({}, [], "", None)]

    else:
        return obj


def serialize_symmetrized_structure(structure: SymmetrizedStructure) -> dict:
    d = structure.as_dict()
    d["equivalent_positions"] = (
        d["equivalent_positions"].tolist()
        if isinstance(d["equivalent_positions"], np.ndarray)
        else d["equivalent_positions"]
    )
    d["spacegroup"] = structure.spacegroup.int_number
    d = remove_empty_fields(d)
    return d


def structure_to_serialized_symmetrized_structure_and_cif(
    structure: Structure,
) -> tuple[dict, str] | tuple[None, None]:
    try:
        analyzer = SpacegroupAnalyzer(structure)
        symmetrized_structure = analyzer.get_symmetrized_structure()

    except ValueError:
        # Symmetry detection can fail for some structures
        return None, None

    symmetrized_structure = serialize_symmetrized_structure(symmetrized_structure)
    cif = analyzer_to_cif(analyzer=analyzer)

    return symmetrized_structure, cif
