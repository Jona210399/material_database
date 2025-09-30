from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupOperations
from pymatgen.symmetry.groups import SpaceGroup
from pymatgen.symmetry.structure import SymmetrizedStructure

from material_database.types_ import SerializedSymmetrizedStructure


def symmetrized_structure_from_serialized(
    entry: SerializedSymmetrizedStructure,
) -> SymmetrizedStructure:
    spacegroup = SpaceGroup.from_int_number(entry["spacegroup"])
    entry["spacegroup"] = SpacegroupOperations(
        int_number=spacegroup.int_number,
        int_symbol=spacegroup.full_symbol,
        symmops=list(spacegroup.symmetry_ops),
    )
    return SymmetrizedStructure.from_dict(entry)


def structure_from_serialized(entry: SerializedSymmetrizedStructure) -> Structure:
    return Structure.from_dict(entry["structure"])
