from typing import TypedDict

import polars as pl
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


class SerializedPXRDPeaks(TypedDict):
    """
    A TypedDict representing the serialized form of PXRD peaks.
    """

    peak_two_thetas: list[float]
    peak_intensities: list[float]

    @classmethod
    def get_schema(cls) -> pl.Schema:
        return pl.Schema(
            {
                "peak_two_thetas": pl.List(pl.Float32),
                "peak_intensities": pl.List(pl.Float32),
            }
        )


class SerializedPXRDGaussianScherrerProfile(TypedDict):
    """
    A TypedDict representing the serialized form of a PXRD pattern.
    """

    intensities: list[float]
    crystallite_size: float

    @classmethod
    def get_schema(cls, intensities_shape: tuple[int, ...]) -> pl.Schema:
        return pl.Schema(
            {
                "intensities": pl.Array(pl.Float32, shape=intensities_shape),
                "crystallite_size": pl.Float32,
            }
        )
