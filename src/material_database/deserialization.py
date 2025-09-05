from pymatgen.symmetry.analyzer import SpacegroupOperations
from pymatgen.symmetry.groups import SpaceGroup
from pymatgen.symmetry.structure import SymmetrizedStructure


def symmetrized_structure_from_serialized(entry: dict) -> SymmetrizedStructure:
    spacegroup = SpaceGroup.from_int_number(entry["spacegroup"])
    entry["spacegroup"] = SpacegroupOperations(
        int_number=spacegroup.int_number,
        int_symbol=spacegroup.full_symbol,
        symmops=list(spacegroup.symmetry_ops),
    )
    return SymmetrizedStructure.from_dict(entry)
