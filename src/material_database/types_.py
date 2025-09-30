from typing import TypedDict

from numpy.typing import ArrayLike


class SerializedSymmetrizedStructure(TypedDict):
    """
    A TypedDict representing the serialized form of a SymmetrizedStructure.
    It also contains '@module' and '@class' fields for compatibility with pymatgen.
    """

    structure: dict
    spacegroup: int
    equivalent_positions: ArrayLike
    wyckoff_letters: list[str]
