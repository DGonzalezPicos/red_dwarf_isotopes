"""
Red Dwarf Isotopes Analysis Package
=================================

A Python package for analyzing isotopic compositions in red dwarf stars through
high-resolution spectroscopy. This package provides tools for:

- Spectral analysis and manipulation
- Radiative transfer modeling
- Chemical composition calculations
- Temperature-pressure profile handling
- Data visualization and plotting

The package is designed for studying molecular isotopologues in M-dwarf atmospheres,
with particular focus on CO, H2O, and other key species.

Classes
-------
Spectrum : Base class for spectral data handling
ModelSpectrum : Class for synthetic spectrum generation
PTProfile : Class for temperature-pressure profile handling
Chemistry : Class for chemical composition calculations
FastChemistry : Class for equilibrium chemistry calculations

Functions
---------
get_data_file : Get path to data file
get_data_dir : Get path to data directory
get_project_root : Get project root directory
ensure_dir_exists : Ensure directory exists

Notes
-----
This package requires petitRADTRANS for radiative transfer calculations.
Some functionality may be limited without the full opacity database.
"""

__version__ = "0.1.0"
__author__ = "Darío González Picos"
__email__ = "picos(at)strw.leidenuniv.nl"
__license__ = "MIT"
__copyright__ = "Copyright 2024 Red Dwarf Isotopes Project Contributors"

# Import core classes
from .spectrum import (
    Spectrum,
    ModelSpectrum,
    DataSpectrum
)

from .PT_profile import (
    get_PT_profile_class,
    PTProfile,
    PTProfileRCE
)

from .chemistry import (
    get_chemistry_class,
    Chemistry,
    FastChemistry
)

# Import utility functions
from .utils import (
    get_data_file,
    get_data_dir,
    get_project_root,
    ensure_dir_exists
)

# Define package exports
__all__ = [
    # Core classes
    "Spectrum",
    "ModelSpectrum",
    "DataSpectrum",
    "PTProfile",
    "PTProfileRCE",
    "Chemistry",
    "FastChemistry",
    
    # Factory functions
    "get_PT_profile_class",
    "get_chemistry_class",
    
    # Utility functions
    "get_data_file",
    "get_data_dir",
    "get_project_root",
    "ensure_dir_exists",
] 