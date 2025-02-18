"""
# Red Dwarf Isotopes

This repository contains supplementary code and data visualization for our research paper on isotopic compositions in red dwarf stars. The code provides tools for manipulating and plotting high-resolution spectroscopic data, with a focus on molecular isotopologues in M-dwarf atmospheres.

Several features of this code are adapted from the package [samderegt/retrieval_base](https://github.com/samderegt/retrieval_base).

## Key Features

- High-resolution spectral analysis tools:
  - Radial velocity corrections
  - Rotational and instrumental broadening
  - Wavelength rebinning and interpolation
  - Continuum normalization
  - Multi-order échelle spectra handling

- Atmospheric modeling:
  - Temperature-pressure profile calculations
  - Chemical equilibrium computations
  - Radiative transfer modeling
  - Isotopologue abundance analysis

- Data handling and visualization:
  - Robust file path management
  - Automated directory creation
  - Publication-quality plotting
  - Error handling and validation

## Installation

```bash
# Clone the repository
git clone https://github.com/DGonzalezPicos/red_dwarf_isotopes.git
cd red_dwarf_isotopes

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install dependencies
pip install -e .
```

## Dependencies

All required Python packages are listed in `requirements.txt` and will be automatically installed with the above commands:

- numpy (≥1.24.0): Array operations and numerical computations
- scipy (≥1.10.0): Scientific computing and interpolation
- petitRADTRANS (≥2.0.0): Radiative transfer calculations
- spectres (≥0.1.0): Spectral resampling
- pyasl (≥3.0.0): Astronomical algorithms
- pyparsing (≥3.0.0): Parsing utilities
- pydantic (≥2.0.0): Data validation

Note: petitRADTRANS installation is required for radiative transfer modeling. See the [petitRADTRANS documentation](https://petitRADTRANS.readthedocs.io/en/2.7.7/content/installation.html) for installation instructions. Line-by-line opacities must be downloaded separately (contact: picos(at)strw.leidenuniv.nl).

## Usage

### Spectral Analysis

```python
from red_dwarf_isotopes import Spectrum, ModelSpectrum

# Load and process observed spectrum
spectrum = Spectrum(wavelength, flux, error)
spectrum.rv_shift(rv=10.0)  # Apply RV correction (km/s)
spectrum.rot_broadening(vsini=5.0)  # Apply rotational broadening (km/s)
spectrum.normalize_flux_per_order()  # Normalize flux

# Create and process model spectrum
model = ModelSpectrum(wavelength, flux, lbl_opacity_sampling=1)
model.shift_broaden_rebin(
    rv=10.0,          # Radial velocity (km/s)
    vsini=5.0,        # Rotational velocity (km/s)
    out_res=70000,    # Output resolution
    d_wave=obs_wave   # Target wavelength grid
)
```

### Atmospheric Structure

```python
from red_dwarf_isotopes import get_PT_profile_class, get_chemistry_class
import numpy as np

# Set up atmospheric grid
pressure = np.logspace(-5, 2, 40)  # bar
temperature = np.linspace(1000, 3000, 40)  # K

# Initialize PT profile
pt_profile = get_PT_profile_class(pressure, mode='RCE')
params = {
    'log_P_RCE': 0.0,
    'dlnT_dlnP_knots': [0.2, 0.15, 0.1, 0.05],
    'T_0': 2500
}
temperature = pt_profile(params)

# Initialize chemistry
species = {
    '12CO': 'CO_high_Sam',
    'H2O': 'H2O_pokazatel_main_iso'
}
chemistry = get_chemistry_class(
    line_species=species,
    pressure=pressure,
    mode='fastchem',
    fastchem_grid_file='path/to/fastchem_grid.h5'
)

# Calculate mass fractions
params = {
    'alpha_12CO': -1.0,  # CO abundance scaling
    'alpha_H2O': -0.2,   # H2O abundance scaling
}
mass_fractions = chemistry(params, temperature)
```

### File Management

```python
from red_dwarf_isotopes import get_data_file, get_data_dir, ensure_dir_exists

# Get paths
data_file = get_data_file('gl15A/bestfit_spec.npy')
output_dir = get_data_dir() / 'outputs'

# Ensure directory exists
ensure_dir_exists(output_dir)
```

## Project Structure

```
red_dwarf_isotopes/
├── data/               # Data files
│   ├── gl338B/        # Individual star directories
│   ├── gl436/
│   └── ...
├── examples/          # Example scripts
│   ├── basic_analysis.py
│   └── plot_bestfit_spectra.py
├── red_dwarf_isotopes/# Main package
│   ├── __init__.py    # Package initialization
│   ├── spectrum.py    # Spectral analysis
│   ├── PT_profile.py  # Temperature-pressure profiles
│   ├── chemistry.py   # Chemical composition
│   ├── pRT_model.py   # Radiative transfer
│   └── utils.py       # Utility functions
├── LICENSE           # MIT License
├── README.md         # This file
├── requirements.txt  # Dependencies
└── setup.py         # Installation
```

## Citation

If you use this code in your research, please cite our paper:

[Paper citation to be added upon publication]

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions about the code or opacity data access:
- Darío González Picos
- Email: picos(at)strw.leidenuniv.nl
- Institution: Leiden Observatory, Leiden University