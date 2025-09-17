import re

from monty.io import zopen
from pymatgen.util.typing import PathLike
from typing_extensions import Self

from material_database.cif.pymatgen.block import CifBlock


class CifFile:
    """Read and parse CifBlocks from a .cif file or string."""

    def __init__(
        self,
        data: dict[str, CifBlock],
        orig_string: str | None = None,
        comment: str | None = None,
    ) -> None:
        """
        Args:
            data (dict): Of CifBlock objects.
            orig_string (str): The original CIF.
            comment (str): Comment.
        """
        self.data = data
        self.orig_string = orig_string
        self.comment: str = comment or "# generated using pymatgen"

    def __str__(self) -> str:
        out = "\n".join(map(str, self.data.values()))
        return f"{self.comment}\n{out}\n"

    @classmethod
    def from_str(cls, string: str) -> Self:
        """Read CifFile from a string.

        Args:
            string: String representation.

        Returns:
            CifFile
        """
        dct = {}

        for block_str in re.split(
            r"^\s*data_", f"x\n{string}", flags=re.MULTILINE | re.DOTALL
        )[1:]:
            # Skip over Cif block that contains powder diffraction data.
            # Some elements in this block were missing from CIF files in
            # Springer materials/Pauling file DBs.
            # This block does not contain any structure information anyway, and
            # CifParser was also not parsing it.
            if "powder_pattern" in re.split(r"\n", block_str, maxsplit=1)[0]:
                continue
            block = CifBlock.from_str(f"data_{block_str}")
            # TODO (@janosh, 2023-10-11) multiple CIF blocks with equal header will overwrite each other,
            # latest taking precedence. maybe something to fix and test e.g. in test_cif_writer_write_file
            dct[block.header] = block

        return cls(dct, string)

    @classmethod
    def from_file(cls, filename: PathLike) -> Self:
        """
        Read CifFile from a filename.

        Args:
            filename: Filename

        Returns:
            CifFile
        """
        with zopen(filename, mode="rt", errors="replace", encoding="utf-8") as file:
            return cls.from_str(file.read())  # type:ignore[arg-type]
