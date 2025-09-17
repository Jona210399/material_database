import numpy as np
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.symmetry.structure import SymmetrizedStructure

from material_database.cif.writing import analyzer_to_cif


def remove_empty_fields(d: dict) -> dict:
    if not isinstance(d, dict):
        return d

    new_dict = {}
    for k, v in d.items():
        if isinstance(v, dict):
            cleaned = remove_empty_fields(v)
            if cleaned:
                new_dict[k] = cleaned
        else:
            new_dict[k] = v
    return new_dict


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
) -> tuple[dict, str]:
    analyzer = SpacegroupAnalyzer(structure)
    symmetrized_structure = analyzer.get_symmetrized_structure()
    symmetrized_structure = serialize_symmetrized_structure(symmetrized_structure)
    cif = analyzer_to_cif(analyzer=analyzer)

    return symmetrized_structure, cif
