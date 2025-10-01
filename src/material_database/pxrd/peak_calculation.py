import json
from pathlib import Path

from xrdsim.calculator import XRDCalculator
from xrdsim.constants import DEFAULT_WAVELENGTH
from xrdsim.numpy.peak_calculator import NumbaXRDPeakCalculator
from xrdsim.numpy.peak_profiles import PeaksOnlyProfile

from material_database.deserialization import structure_from_serialized
from material_database.types_ import SerializedPXRDPeaks, SerializedSymmetrizedStructure


def get_peak_calculation_calculator():
    return XRDCalculator(
        peak_calculator=NumbaXRDPeakCalculator(
            wavelength=DEFAULT_WAVELENGTH,
            angle_range=(0.0, 180.0),
        ),
        peak_profile=PeaksOnlyProfile(),
        rescale_intensity=False,
    )


def writeout_constant_metadata(destination: Path):
    calculator = get_peak_calculation_calculator()
    meta_data = calculator.get_constant_metadata()

    with open(destination / "metadata.json", "w") as f:
        json.dump(meta_data, f)


def calculate_peaks_from_symmetrized_structure_entry(
    entry: SerializedSymmetrizedStructure,
) -> SerializedPXRDPeaks:
    structure = structure_from_serialized(entry)

    calculator = get_peak_calculation_calculator()

    two_thetas, intensities, _ = calculator.calculate(structure)

    return SerializedPXRDPeaks(
        peak_two_thetas=two_thetas.tolist(),
        peak_intensities=intensities.tolist(),
    )
