import json
from pathlib import Path

import numpy as np
from xrdsim.constants import (
    DEFAULT_ANGLE_RANGE,
    DEFAULT_CRYSTALLITE_SIZE_RANGE,
    DEFAULT_SHAPEFACTOR,
    DEFAULT_WAVELENGTH,
)
from xrdsim.numpy.crystallite_size import UniformCrystalliteSampler
from xrdsim.numpy.peak_profiles import GaussianProfile, GaussianScherrerProfile

from material_database.types_ import SerializedPXRDGaussianScherrerProfile


def construct_peak_profile():
    return GaussianScherrerProfile(
        gaussian_profile=GaussianProfile(DEFAULT_ANGLE_RANGE),
        shape_factor=DEFAULT_SHAPEFACTOR,
        wavelength=DEFAULT_WAVELENGTH,
        crystallite_size_provider=UniformCrystalliteSampler(
            DEFAULT_CRYSTALLITE_SIZE_RANGE
        ),
    )


def writeout_constant_metadata(destination: Path):
    peak_profile = construct_peak_profile()
    meta_data = peak_profile.get_constant_metadata()

    with open(destination / "metadata.json", "w") as f:
        json.dump(meta_data, f)


def convolve_serialized_peaks(
    peak_two_thetas: list[float],
    peak_intensities: list[float],
) -> SerializedPXRDGaussianScherrerProfile:
    peak_profile = construct_peak_profile()
    peak_two_thetas = np.array(peak_two_thetas)
    peak_intensities = np.array(peak_intensities)

    angle_range = peak_profile.gaussian_profile.x_range

    mask = (peak_two_thetas >= angle_range[0]) & (peak_two_thetas <= angle_range[1])

    peak_two_thetas = peak_two_thetas[mask]
    peak_intensities = peak_intensities[mask]

    _, intensities = peak_profile.convolute_peaks(peak_two_thetas, peak_intensities)

    metadata = peak_profile.get_metadata()

    max_intensity = np.nanmax(intensities)

    if not np.isfinite(max_intensity) or max_intensity <= 0:
        return SerializedPXRDGaussianScherrerProfile(
            intensities=None,
            crystallite_size=None,
        )

    intensities = intensities / np.max(intensities)

    return SerializedPXRDGaussianScherrerProfile(
        intensities=intensities.tolist(),
        crystallite_size=metadata["crystallite_size"],
    )
