from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

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


def structure_to_serialized_symmetrized_structure_and_cif(
    structure,
) -> tuple[dict, str]:
    analyzer = SpacegroupAnalyzer(structure)
    symmetrized_structure = analyzer.get_symmetrized_structure()
    symmetrized_structure = symmetrized_structure.as_dict()
    symmetrized_structure["equivalent_positions"] = symmetrized_structure[
        "equivalent_positions"
    ].tolist()
    symmetrized_structure = remove_empty_fields(symmetrized_structure)
    symmetrized_structure["spacegroup"] = analyzer.get_space_group_number()
    cif = analyzer_to_cif(analyzer=analyzer)

    return symmetrized_structure, cif
