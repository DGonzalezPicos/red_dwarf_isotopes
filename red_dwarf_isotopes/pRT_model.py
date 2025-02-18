"""
Radiative transfer modeling using petitRADTRANS.

This module provides a wrapper around petitRADTRANS for generating synthetic spectra
of red dwarf stars. It handles the setup of atmospheric structures, opacity sources,
and spectral calculations.

Note
----
This is a read-only script. To generate model spectra, you need:
1. petitRADTRANS v2.7.7 installation
2. Line-by-line opacity data (>100 GB)

For opacity data access, contact: picos(at)strw.leidenuniv.nl
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import numpy.typing as npt

from petitRADTRANS import Radtrans
import petitRADTRANS.nat_cst as nc

from .spectrum import DataSpectrum, ModelSpectrum
from retrieval_base.auxiliary_functions import apply_PT_cutoff

ArrayType = npt.NDArray[np.float64]

class PRTModel:
    """
    petitRADTRANS model wrapper for generating synthetic stellar spectra.
    
    This class handles the setup and computation of radiative transfer models
    using petitRADTRANS, including atmospheric structure, opacities, and
    spectral synthesis.
    
    Attributes
    ----------
    d_wave : ArrayType
        Wavelength grid from observed data
    d_resolution : float
        Spectral resolution of observed data
    line_species : List[str]
        Species to include in line opacity calculations
    mode : str
        petitRADTRANS mode ('lbl' or 'c-k')
    pressure : ArrayType
        Atmospheric pressure grid
    atm : List[Radtrans]
        petitRADTRANS atmosphere objects for each spectral order
    """

    def __init__(
        self, 
        line_species: List[str],
        data_spectrum: 'DataSpectrum',
        mode: str = 'lbl',
        lbl_opacity_sampling: int = 3,
        rayleigh_species: List[str] = ['H2', 'He'],
        continuum_opacities: List[str] = ['H2-H2', 'H2-He'],
        pressure: Optional[ArrayType] = None,
        log_P_range: Tuple[float, float] = (-5, 2),
        n_atm_layers: int = 40,
        rv_range: Tuple[float, float] = (-50, 50),
        T_cutoff: Optional[Tuple[float, float]] = None,
        P_cutoff: Optional[Tuple[float, float]] = None,
    ) -> None:
        """
        Initialize PRTModel.

        Parameters
        ----------
        line_species : List[str]
            List of species to include in line opacity calculations
        data_spectrum : DataSpectrum
            Observed spectrum object containing wavelength grid and properties
        mode : str, optional
            petitRADTRANS mode ('lbl' or 'c-k'), by default 'lbl'
        lbl_opacity_sampling : int, optional
            Sampling rate for line-by-line calculations, by default 3
        rayleigh_species : List[str], optional
            Species for Rayleigh scattering, by default ['H2', 'He']
        continuum_opacities : List[str], optional
            Species for continuum opacity, by default ['H2-H2', 'H2-He']
        pressure : Optional[ArrayType], optional
            Custom pressure grid, by default None
        log_P_range : Tuple[float, float], optional
            Log pressure range [min, max], by default (-5, 2)
        n_atm_layers : int, optional
            Number of atmospheric layers, by default 40
        rv_range : Tuple[float, float], optional
            Radial velocity range [min, max] in km/s, by default (-50, 50)
        T_cutoff : Optional[Tuple[float, float]], optional
            Temperature cutoff range for custom opacities, by default None
        P_cutoff : Optional[Tuple[float, float]], optional
            Pressure cutoff range for custom opacities, by default None
        """
        # Store data properties
        self.d_wave = data_spectrum.wave
        self.d_mask_isfinite = data_spectrum.mask_isfinite
        self.d_resolution = data_spectrum.resolution
        self.w_set = data_spectrum.w_set

        # Store model parameters
        self.line_species = line_species
        self.mode = mode
        self.lbl_opacity_sampling = lbl_opacity_sampling
        self.rayleigh_species = rayleigh_species
        self.continuum_species = continuum_opacities
        
        # Temperature and pressure cutoffs for custom opacities
        self.T_cutoff = T_cutoff
        self.P_cutoff = P_cutoff

        # Set maximum RV for wavelength padding
        self.rv_max = max(40.0, max(abs(rv_range[0]), abs(rv_range[1])))

        # Set up atmospheric pressure grid
        if pressure is None:
            self.pressure = np.logspace(log_P_range[0], log_P_range[1], n_atm_layers)
        else:
            self.pressure = np.array(pressure)

        # Initialize petitRADTRANS atmospheres
        self._setup_atmospheres()

    def _setup_atmospheres(self) -> None:
        """
        Initialize petitRADTRANS atmosphere objects for each spectral order.
        
        This creates separate Radtrans objects for each order to handle
        different wavelength ranges efficiently.
        """
        # Calculate wavelength ranges with padding for RV shifts
        wave_pad = 1.1 * self.rv_max/(nc.c*1e-5) * self.d_wave.max()
        self.wave_range_micron = np.vstack([
            self.d_wave.min(axis=(1,2)) - wave_pad,
            self.d_wave.max(axis=(1,2)) + wave_pad
        ]).T * 1e-3  # Convert to microns

        # Create atmosphere objects for each order
        self.atm = []
        for i, wave_range in enumerate(self.wave_range_micron):
            print(f'Loading opacities for {wave_range[0]:.2f}-{wave_range[1]:.2f} μm')
            
            # Initialize Radtrans object
            atm = Radtrans(
                line_species=self.line_species,
                rayleigh_species=self.rayleigh_species,
                continuum_opacities=self.continuum_species,
                wlen_bords_micron=wave_range,
                mode=self.mode,
                lbl_opacity_sampling=self.lbl_opacity_sampling,
            )

            # Set up atmospheric structure
            atm.setup_opa_structure(self.pressure)
            
            # Apply temperature-pressure cutoffs if specified
            if self.T_cutoff is not None:
                P_range = self.P_cutoff or (self.pressure.min(), self.pressure.max())
                atm = apply_PT_cutoff(atm, *self.T_cutoff, *P_range)
                
            self.atm.append(atm)

    def __call__(
        self,
        mass_fractions: Dict[str, ArrayType],
        temperature: ArrayType,
        params: Dict[str, Union[float, ArrayType]]
    ) -> ModelSpectrum:
        """
        Generate a synthetic spectrum with given parameters.

        Parameters
        ----------
        mass_fractions : Dict[str, ArrayType]
            Mass fractions for each species
        temperature : ArrayType
            Temperature profile
        params : Dict[str, Union[float, ArrayType]]
            Model parameters including:
            - log_g: Surface gravity
            - rv: Radial velocity (km/s)
            - vsini: Projected rotational velocity (km/s)
            - epsilon_limb: Limb darkening coefficient
            - resolution: Spectral resolution (optional)
            - gamma: Lorentzian component for Voigt profile (optional)
            - fwhm: FWHM for Voigt profile (optional)

        Returns
        -------
        ModelSpectrum
            Synthetic spectrum object
        """
        self.mass_fractions = mass_fractions
        self.temperature = temperature
        self.params = params
        
        return self._compute_spectrum()

    def _compute_spectrum(self) -> ModelSpectrum:
        """
        Compute synthetic spectrum for all spectral orders.

        Returns
        -------
        ModelSpectrum
            Combined synthetic spectrum for all orders
        """
        wave = np.ones_like(self.d_wave) * np.nan
        flux = np.ones_like(self.d_wave) * np.nan
        
        # Process each order
        for i, atm in enumerate(self.atm):
            # Compute radiative transfer
            atm.calc_flux(
                self.temperature,
                self.mass_fractions,
                gravity=10.0**self.params['log_g'],
                mmw=self.mass_fractions['MMW']
            )
            
            # Convert wavelength and flux
            wave_i = nc.c / atm.freq * 1e7  # Convert to nm
            flux_i = atm.flux * nc.c / (wave_i/1e7)**2  # Convert to per wavelength
            
            # Handle numerical overflow
            flux_i = np.where(np.log(flux_i) <= 20, flux_i, 0.0)
            
            # Convert units
            flux_i *= 1e-7  # erg/cm²/s/nm
            
            # Apply stellar radius scaling if needed
            R_p = self.params.get('R_p', 0.0)
            if R_p > 0:
                flux_i *= ((R_p*nc.r_jup_mean) / 
                          (1e3/self.params['parallax']*nc.pc))**2

            # Create and process model spectrum
            model = ModelSpectrum(
                wave=wave_i,
                flux=flux_i,
                lbl_opacity_sampling=self.lbl_opacity_sampling
            )
            
            # Apply spectral operations
            model.shift_broaden_rebin(
                rv=self.params['rv'],
                vsini=self.params['vsini'],
                epsilon_limb=self.params['epsilon_limb'],
                out_res=self.params.get('resolution', self.d_resolution),
                in_res=model.resolution,
                rebin=False,
                gamma=self.params.get('gamma'),
                fwhm=self.params.get('fwhm')
            )

            # Rebin to data wavelength grid
            model.rebin(d_wave=self.d_wave[i,:], replace_wave_flux=True)
            
            # Store results
            wave[i,:,:] = model.wave
            flux[i,:,:] = model.flux

        # Create combined spectrum
        return ModelSpectrum(
            wave=wave,
            flux=flux,
            lbl_opacity_sampling=self.lbl_opacity_sampling
        )



