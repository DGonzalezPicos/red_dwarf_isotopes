# Red Dwarf Isotopes

This repository contains supplementary code and data visualization for our research paper on isotopic compositions in red dwarf stars. The code provides tools for manipulating and plotting high-resolution spectroscopic data. It also includes a basic example of the radiative transfer model used to compute the synthetic spectra.

Several features of this code are adapted from the package [samderegt/retrieval_base](https://github.com/samderegt/retrieval_base).

### Key Features

- Spectral analysis tools for high-resolution stellar spectra
- Handling of multi-order échelle spectra
- Spectral operations:
  - Radial velocity corrections
  - Rotational and instrumental broadening
  - Wavelength rebinning and interpolation
  - Continuum normalization

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

All required Python packages are listed in `requirements.txt` and will be automatically installed with the above commands. Installation of `petitRADTRANS` is optional, please see the [petitRADTRANS](https://petitRADTRANS.readthedocs.io/en/2.7.7/content/installation.html) for more information. Note that generating radiative transfer models requires line-by-line opacities to be downloaded separately (see the [petitRADTRANS documentation](https://petitradtrans.readthedocs.io/en/2.7.7/content/installation.html#before-installation-download-the-opacity-data) or contact the authors).

## Usage

```python
from red_dwarf_isotopes import Spectrum

# Load and process a spectrum
spectrum = Spectrum(wavelength, flux, error)

# Apply radial velocity correction
spectrum.rv_shift(rv=10.0)  # km/s

# Apply rotational broadening
spectrum.rot_broadening(vsini=5.0)  # km/s

# Normalize the spectrum
spectrum.normalize_flux_per_order()
```

### Model Spectrum Operations

```python
from red_dwarf_isotopes import ModelSpectrum

# Create a model spectrum
model = ModelSpectrum(wavelength, flux, lbl_opacity_sampling=1)

# Combined operations (RV shift, broadening, and rebinning)
model.shift_broaden_rebin(
    rv=10.0,          # Radial velocity in km/s
    vsini=5.0,        # Rotational velocity in km/s
    out_res=70000,    # Output spectral resolution
    d_wave=obs_wave   # Target wavelength grid
)
```

### Example Scripts

The `examples/` directory contains several example scripts demonstrating the analysis pipeline:

- `basic_analysis.py`: Basic spectrum processing
- `plot_bestfit_spectra.py`: Plotting observed and model spectra

## Data Structure

```
red_dwarf_isotopes/
├── data/               # Data files
│   ├── gl338B/        # Individual star directories
│   ├── gl436/
│   └── ...
├── examples/          # Example notebooks and scripts
├── red_dwarf_isotopes/# Main package directory
│   ├── spectrum.py    # Spectral analysis 
│   ├── pRT_model.py   # Radiative transfer model
├── LICENSE           # MIT License
├── README.md         # This file
├── requirements.txt  # Package dependencies
└── setup.py         # Package installation script
```

## Citation

If you use this code in your research, please cite our paper:


## License

This project is licensed under the MIT License - see the LICENSE file for details.