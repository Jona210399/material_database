import re
from logging import WARNING, getLogger

from pymatgen.core import DummySpecies, Element, Species, get_el_sp
from pymatgen.io.core import ParseError

from material_database.cif.parsing.block import CifBlock

LOGGER = getLogger(__name__)
LOGGER.setLevel(WARNING)


def get_species_from_atom_site(
    atom_site: str,
    oxidation_states: dict[str, float] | None,
) -> Element | Species | DummySpecies | None:
    element_symbol = _parse_atom_site(atom_site)

    if not element_symbol:
        return None

    if oxidation_states is not None:
        oxidation_state = oxidation_states.get(element_symbol, 0)
        oxidation_state = oxidation_states.get(atom_site, oxidation_state)

        try:
            return Species(element_symbol, oxidation_state)
        except (ValueError, ParseError):
            return DummySpecies(element_symbol, oxidation_state)

    return get_el_sp(element_symbol)


def parse_atom_sites(block: CifBlock) -> list[str] | None:
    atom_site_labels = block.get("_atom_site_label", [])
    atom_site_type_symbols = block.get("_atom_site_type_symbol", None)
    atom_sites = atom_site_type_symbols or atom_site_labels
    return atom_sites


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
