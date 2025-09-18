from material_database.cif.parsing.block import CifBlock


def get_wyckoff_letters_as_in_cif(block: CifBlock) -> list[str] | None:
    KEYS_TO_TRY = [
        "_atom_site_Wyckoff_symbol",
        "_atom_site_Wyckoff_symbol_",
    ]
    for key in KEYS_TO_TRY:
        wyckoff_letters: list[str] = block.get(key)
        if wyckoff_letters:
            return [letter.strip() for letter in wyckoff_letters]

    return None


def wyckoff_multiplicity_str_to_int(s: str) -> int | None:
    if s.strip().isnumeric():
        return int(s.strip())
    return None


def get_wyckoff_multiplicities(block: CifBlock) -> list[int] | None:
    KEYS_TO_TRY = [
        "_atom_site_Wyckoff_multiplicity",
        "_atom_site_Wyckoff_multiplicity_",
        "_atom_site_symmetry_multiplicity",
        "_atom_site_symmetry_multiplicity_",
    ]

    for key in KEYS_TO_TRY:
        multiplicities: list[str] = block.get(key)
        if multiplicities:
            multiplicities = [
                wyckoff_multiplicity_str_to_int(mult) for mult in multiplicities
            ]

            if None not in multiplicities:
                return multiplicities

    return None


def get_wyckoff_symbols(multiplicities: list[int], letters: list[str]) -> list[str]:
    if len(multiplicities) != len(letters):
        raise ValueError("Multiplicity and letters lists must have the same length.")

    return [f"{mult}{letter}" for mult, letter in zip(multiplicities, letters)]


def get_wyckoff_letters(
    wyckoff_letters_as_in_cif: list[str], multiplicities: list[int]
) -> list[str]:
    wyckoff_letters = []
    for letter, count in zip(wyckoff_letters_as_in_cif, multiplicities):
        wyckoff_letters.extend([letter] * count)
    return wyckoff_letters


def parse_wyckoff_info(block: CifBlock):
    wyckoff_letters_as_in_cif = get_wyckoff_letters_as_in_cif(block)
    wyckoff_multiplicities = get_wyckoff_multiplicities(block)

    if wyckoff_letters_as_in_cif is None:
        raise ValueError("Could not determine Wyckoff letters from CIF block.")

    if wyckoff_multiplicities is None:
        raise ValueError("Could not determine Wyckoff multiplicities from CIF block.")

    if len(wyckoff_letters_as_in_cif) != len(wyckoff_multiplicities):
        raise ValueError("Length of wyckoff letters must match multiplicities")

    return get_wyckoff_letters(wyckoff_letters_as_in_cif, wyckoff_multiplicities)


def get_equivalent_positions(multiplicities: list[int]) -> list[int]:
    equivalent_positions = []
    for group_index, count in enumerate(multiplicities):
        equivalent_positions.extend([group_index] * count)
    return equivalent_positions
