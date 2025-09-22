import re

from pymatgen.core import Composition, DummySpecies, Element, Species
from pymatgen.io.core import ParseError

from material_database.cif.parsing.block import CifBlock
from material_database.cif.parsing.logger import LOGGER as PARSER_LOGGER
from material_database.cif.parsing.utils import str2float

LOGGER = PARSER_LOGGER.getChild("atomic_sites")
MIN_OCCUPANCY = 1e-8


def parse_compositions(
    block: CifBlock,
) -> tuple[list[Composition], list[str], dict[str, list[int]]] | None:
    atom_site_labels = block.get("_atom_site_label", [])
    atom_site_type_symbols = block.get("_atom_site_type_symbol", None)
    atom_sites = atom_site_type_symbols or atom_site_labels

    if not atom_sites:
        LOGGER.warning("No atom site data found in CIF block.")
        return None

    element_symbols = [_parse_atom_site(site) for site in atom_sites]

    if None in element_symbols:
        LOGGER.warning("Could not parse some atom site labels.")
        return None

    oxidation_states = _parse_oxidation_states(block)
    if oxidation_states is None:
        LOGGER.warning("No oxidation state information found, assuming 0 for all.")
        oxidation_states = {}

    species = [
        _get_species_from_atom_site(site, element, oxidation_states)
        for site, element in zip(atom_sites, element_symbols)
    ]

    implicit_hydrogens = [_get_num_implicit_hydrogens(site) for site in atom_sites]

    if any(num_h > 0 for num_h in implicit_hydrogens):
        LOGGER.warning(
            "Structure has implicit hydrogens defined, parsed structure unlikely to be "
            "suitable for use in calculations unless hydrogens added."
        )

    occupancies = _parse_occupancies(block)

    if occupancies is None:
        LOGGER.warning("Could not parse occupancies from CIF block. Setting all to 1.0")
        occupancies = [1.0] * len(species)

    if None in occupancies:
        LOGGER.warning("Some occupancies could not be parsed. Setting them to 1.0")
        occupancies = [
            occupancy if occupancy is not None else 1.0 for occupancy in occupancies
        ]

    compositions_and_implicit_hydrogens = [
        _get_composition_from_species_and_occupancy(species, occupancy, num_h)
        for species, occupancy, num_h in zip(species, occupancies, implicit_hydrogens)
    ]

    compositions, implicit_hydrogens = zip(*compositions_and_implicit_hydrogens)

    site_properties: dict[str, list[int]] = {}
    if any(implicit_hydrogens):
        LOGGER.warning(
            "Some implicit hydrogens were added to compositions as the site was "
            "not oxygen."
        )
        site_properties["implicit_hydrogens"] = list(implicit_hydrogens)

    return compositions, atom_sites, site_properties


def _parse_occupancies(block: CifBlock) -> list[float | None] | None:
    KEYS_TO_TRY = [
        "_atom_site_occupancy",
        "_atom_site_occupancies",
        "_atom_site_occupancy_ ",
        "_atom_site_occupancies_",
    ]

    for key in KEYS_TO_TRY:
        occupancies: list[str] = block.get(key, None)
        if occupancies is not None:
            break
    else:
        return None

    occupancies_as_floats = []
    for occupancy in occupancies:
        try:
            occupancy = str2float(occupancy.strip())
        except ValueError:
            occupancy = None

        occupancies_as_floats.append(occupancy)

    return occupancies_as_floats


def _parse_oxidation_states(cif_block: CifBlock) -> dict[str, float] | None:
    KEYS_TO_TRY = [
        "_atom_type_oxidation_number",
        "_atom_type_oxidation_number_",
    ]

    for key in KEYS_TO_TRY:
        oxidation_numbers: list[str] = cif_block.get(key, None)
        if oxidation_numbers is not None:
            break
    else:
        return None

    oxidation_numbers_as_floats = []
    for oxidation_number in oxidation_numbers:
        try:
            oxidation_number = str2float(oxidation_number.strip())
            oxidation_numbers_as_floats.append(oxidation_number)
        except ValueError:
            return None

    KEYS_TO_TRY = [
        "_atom_type_symbol",
        "_atom_type_symbol_",
    ]

    for key in KEYS_TO_TRY:
        atom_type_symbols: list[str] = cif_block.get(key, None)
        if atom_type_symbols is not None:
            break
    else:
        return None

    if len(atom_type_symbols) != len(oxidation_numbers_as_floats):
        return None

    oxidation_states = {
        atom_type: oxidation
        for atom_type, oxidation in zip(atom_type_symbols, oxidation_numbers_as_floats)
    }

    atom_type_symbols_stripped = [
        re.sub(r"\d?[\+,\-]?$", "", symbol) for symbol in atom_type_symbols
    ]

    if len(atom_type_symbols_stripped) != len(oxidation_numbers_as_floats):
        return oxidation_states

    oxidation_states.update(
        {
            atom_type: oxidation
            for atom_type, oxidation in zip(
                atom_type_symbols_stripped, oxidation_numbers_as_floats
            )
        }
    )
    return oxidation_states


def _get_composition_from_species_and_occupancy(
    species: Element | Species | DummySpecies,
    occupancy: float,
    num_implicit_hydrogens: int,
) -> tuple[Composition, int]:
    composition_dict: dict[Species | str, float] = {
        species: max(occupancy, MIN_OCCUPANCY)
    }

    if species == Element("O"):
        return Composition(composition_dict), num_implicit_hydrogens

    if num_implicit_hydrogens > 0:
        composition_dict["H"] = num_implicit_hydrogens
        num_implicit_hydrogens = 0

    return Composition(composition_dict), num_implicit_hydrogens


def _get_species_from_atom_site(
    atom_site: str,
    element_symbol: str,
    oxidation_states: dict[str, float],
) -> Element | Species | DummySpecies:
    oxidation_state = oxidation_states.get(element_symbol, 0)
    oxidation_state = oxidation_states.get(atom_site, oxidation_state)

    try:
        return Species(element_symbol, oxidation_state)
    except (ValueError, ParseError):
        return DummySpecies(element_symbol, oxidation_state)


def _parse_atom_site(atom_site: str) -> str | None:
    """Parse a string with a symbol to extract a string representing an element.

    Args:
        sym (str): A symbol to be parsed.

    Returns:
        A string for the parsed symbol. None if no parsing was possible.
    """
    # Common representations for elements/water in CIF files
    # TODO: fix inconsistent handling of water
    special_syms = {
        "Hw": "H",
        "Ow": "O",
        "Wat": "O",
        "wat": "O",
        "OH": "",
        "OH2": "",
        "NO3": "N",
    }

    parsed_sym = None
    # Try with special symbols, otherwise check the first two letters,
    # then the first letter alone. If everything fails try extracting the
    # first letter.
    m_sp = re.match("|".join(special_syms), atom_site)
    if m_sp:
        parsed_sym = special_syms[m_sp.group()]
    elif Element.is_valid_symbol(atom_site[:2].title()):
        parsed_sym = atom_site[:2].title()
    elif Element.is_valid_symbol(atom_site[0].upper()):
        parsed_sym = atom_site[0].upper()
    elif match := re.match(r"w?[A-Z][a-z]*", atom_site):
        parsed_sym = match.group()

    if parsed_sym is not None and (
        m_sp or not re.match(rf"{parsed_sym}\d*", atom_site)
    ):
        msg = f"{atom_site} parsed as {parsed_sym}"
        LOGGER.warning(msg)

    return parsed_sym


def _get_num_implicit_hydrogens(atom_site: str) -> int:
    """Get number of implicit hydrogens."""
    num_h = {"Wat": 2, "wat": 2, "O-H": 1}
    return num_h.get(atom_site[:3], 0)
