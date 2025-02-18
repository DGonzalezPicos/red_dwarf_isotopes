"""
Basic example of analyzing red dwarf spectra for isotopic compositions.
"""

import numpy as np
import matplotlib.pyplot as plt
from red_dwarf_isotopes import process_spectrum
from red_dwarf_isotopes.utils import get_data_file, get_data_dir, ensure_dir_exists

# Load data using robust path handling
try:
    # Try to load existing data
    data_file = get_data_file('gl15A/bestfit_spec.npy')
    data = np.load(data_file)
    wavelengths = data['wavelengths']
    flux = data['flux']
except FileNotFoundError:
    # Generate example data if file doesn't exist
    print("Example data file not found, generating synthetic data...")
    wavelengths = np.linspace(6000, 7000, 1000)  # Wavelength in Angstroms
    flux = np.random.normal(1, 0.1, size=1000)   # Normalized flux
    
    # Add some absorption features
    for center in [6300, 6500, 6800]:
        flux *= 1 - 0.5 * np.exp(-(wavelengths - center)**2 / (2 * 5**2))

# Process the spectrum
result = process_spectrum(wavelengths, flux)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(wavelengths, flux, 'k-', label='Observed')
plt.xlabel('Wavelength (Å)')
plt.ylabel('Normalized Flux')
plt.title('Example Red Dwarf Spectrum')
plt.legend()
plt.grid(True)

# Save plot with robust path handling
output_dir = get_data_dir() / 'outputs'
ensure_dir_exists(output_dir)
plt.savefig(output_dir / 'example_spectrum.png', dpi=300, bbox_inches='tight')
plt.close()

# Save the results with robust path handling
np.savez(output_dir / 'example_results.npz',
         wavelengths=wavelengths,
         flux=flux,
         processed_data=result) 