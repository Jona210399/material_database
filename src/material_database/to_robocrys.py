import warnings
from logging import getLogger

from robocrys import StructureCondenser, StructureDescriber

from material_database.deserialization import structure_from_serialized
from material_database.types_ import SerializedSymmetrizedStructure

STRUCTURE_CONDENSER = StructureCondenser()
STRUCTURE_DESCRIBER = StructureDescriber()

LOGGER = getLogger(__name__)


def generate_text(entry: SerializedSymmetrizedStructure) -> str | None:
    structure = structure_from_serialized(entry)

    try:
        condensed = STRUCTURE_CONDENSER.condense_structure(structure)
        description = STRUCTURE_DESCRIBER.describe(condensed)
    except ValueError as e:
        LOGGER.warning(f"Failed to generate robocrys description: {e}")
        return None

    return description


def mute_warnings():
    warnings.filterwarnings("ignore", category=UserWarning, module="pymatgen")
    warnings.filterwarnings("ignore", category=UserWarning, module="matminer")
    warnings.filterwarnings("ignore", category=RuntimeWarning, module="robocrys")
