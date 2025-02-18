"""Generate a model spectrum with petitRADTRANS given a set of parameters.

WARNING: this is a read-only script, to generate model spectra you need the opacities
and the petitRADTRANS v2.7.7 installation.

If you wish to obtain the opacities (>100 GB), please contact picos(at)strw.leidenuniv.nl
"""

import numpy as np

from petitRADTRANS import Radtrans
import petitRADTRANS.nat_cst as nc

from .spectrum import ModelSpectrum
from retrieval_base.auxiliary_functions import apply_PT_cutoff

class pRT_model:

    def __init__(self, 
                 line_species, 
                 d_spec, 
                 mode='lbl', 
                 lbl_opacity_sampling=3, 
                 rayleigh_species=['H2', 'He'], 
                 continuum_opacities=['H2-H2', 'H2-He'], 
                 log_P_range=(-5,2), 
                 n_atm_layers=40, 
                 pressure=None,
                 chem_mode='free', 
                 rv_range=(-50,50),
                 T_cutoff=None,
                 P_cutoff=None,
                 ):
        '''
        Create instance of the pRT_model class.

        Input
        -----
        line_species : list
            Names of line-lists to include.
        d_spec : DataSpectrum
            Instance of the DataSpectrum class.
        mode : str
            pRT mode to use, can be 'lbl' or 'c-k'.
        lbl_opacity_sampling : int
            Let pRT sample every n-th datapoint.
        cloud_species : list or None
            Chemical cloud species to include. 
        rayleigh_species : list
            Rayleigh-scattering species.
        continuum_opacities : list
            CIA-induced absorption species.
        log_P_range : tuple or list
            Logarithm of modelled pressure range.
        n_atm_layers : int
            Number of atmospheric layers to model.
        chem_mode : str
            Chemistry mode to use for clouds, can be 'free' or 'eqchem'.
        
        '''

        # Read in attributes of the observed spectrum
        self.d_wave          = d_spec.wave
        self.d_mask_isfinite = d_spec.mask_isfinite
        self.d_resolution    = d_spec.resolution
        self.apply_high_pass_filter = d_spec.high_pass_filtered
        self.w_set = d_spec.w_set

        self.line_species = line_species
        self.mode = mode
        self.lbl_opacity_sampling = lbl_opacity_sampling

        self.rayleigh_species  = rayleigh_species
        self.continuum_species = continuum_opacities
        
        self.T_cutoff = T_cutoff # temperature cutoff for custom line opacities
        self.P_cutoff = P_cutoff # pressure cutoff for custom line opacities

        self.chem_mode  = chem_mode

        self.rv_max = max(40.0, max(np.abs(list(rv_range))))

        # Define the atmospheric layers
        if log_P_range is None:
            log_P_range = (-5,2)
        if n_atm_layers is None:
            n_atm_layers = 40
            
        if pressure is None:
            self.pressure = np.logspace(log_P_range[0], log_P_range[1], n_atm_layers)
        else:
            self.pressure = np.array(pressure)

        # Make the pRT.Radtrans objects
        self.get_atmospheres()

    def get_atmospheres(self):

        # pRT model is somewhat wider than observed spectrum
        wave_pad = 1.1 * self.rv_max/(nc.c*1e-5) * self.d_wave.max()

        self.wave_range_micron = np.concatenate(
            (self.d_wave.min(axis=(1,2))[None,:]-wave_pad, 
             self.d_wave.max(axis=(1,2))[None,:]+wave_pad
            )).T
        self.wave_range_micron *= 1e-3

        self.atm = []
        for wave_range_i in self.wave_range_micron:
            print(f' Loading opacities for wavelength range {wave_range_i[0]:.2f}-{wave_range_i[1]:.2f} micron')
            # Make a pRT.Radtrans object
            atm_i = Radtrans(
                line_species=self.line_species, 
                rayleigh_species=self.rayleigh_species, 
                continuum_opacities=self.continuum_species, 
                wlen_bords_micron=wave_range_i, 
                mode=self.mode, 
                lbl_opacity_sampling=self.lbl_opacity_sampling, 
                )

            # Set up the atmospheric layers
            atm_i.setup_opa_structure(self.pressure)
            if self.T_cutoff is not None:
                self.P_cutoff = getattr(self, 'P_cutoff', (np.min(self.pressure), np.max(self.pressure)))
                atm_i = apply_PT_cutoff(atm_i, *self.T_cutoff, *self.P_cutoff)
            self.atm.append(atm_i)

    def __call__(self, 
                 mass_fractions, 
                 temperature, 
                 params, 
                 ):
        '''
        Create a new model spectrum with the given arguments.

        Input
        -----
        mass_fractions : dict
            Species' mass fractions in the pRT format.
        temperature : np.ndarray
            Array of temperatures at each atmospheric layer.
        params : dict
            Parameters of the current model.
    
        Returns
        -------
        m_spec : ModelSpectrum class
            Instance of the ModelSpectrum class. 
        '''

        # Update certain attributes
        self.mass_fractions = mass_fractions
        self.temperature    = temperature
        self.params = params

        # Generate a model spectrum
        m_spec = self.get_model_spectrum()
        return m_spec

    def get_model_spectrum(self):
        '''
        Generate a model spectrum with the given parameters.

        Returns
        -------
        m_spec : ModelSpectrum class
            Instance of the ModelSpectrum class
        '''

        # Loop over all orders
        wave = np.ones_like(self.d_wave) * np.nan
        flux = np.ones_like(self.d_wave) * np.nan
        
        for i, atm_i in enumerate(self.atm):
            
            # Compute the emission spectrum
            atm_i.calc_flux(
                self.temperature, 
                self.mass_fractions, 
                gravity=10.0**self.params['log_g'], 
                mmw=self.mass_fractions['MMW'], 
                )
            wave_i = nc.c / atm_i.freq
            flux_i = np.where(np.isfinite(atm_i.flux), atm_i.flux, 0.0)
            overflow = np.log(atm_i.flux) > 20
            atm_i.flux[overflow] = 0.0
            
            flux_i = atm_i.flux *  nc.c / (wave_i**2)

            # Convert [erg cm^{-2} s^{-1} cm^{-1}] -> [erg cm^{-2} s^{-1} nm^{-1}]
            flux_i = flux_i * 1e-7

            # Convert [cm] -> [nm]
            wave_i *= 1e7

            # Convert to observation by scaling with planetary radius
            R_p = getattr(self.params, 'R_p', 0.0)
            if R_p > 0:
                flux_i *= (
                    (R_p*nc.r_jup_mean) / \
                    (1e3/self.params['parallax']*nc.pc)
                    )**2

            # Create a ModelSpectrum instance
            m_spec_i = ModelSpectrum(
                wave=wave_i, flux=flux_i, 
                lbl_opacity_sampling=self.lbl_opacity_sampling
                )
            
            # Apply radial-velocity shift, rotational/instrumental broadening
            m_spec_i.shift_broaden_rebin(
                rv=self.params['rv'], 
                vsini=self.params['vsini'], 
                epsilon_limb=self.params['epsilon_limb'], 
                out_res=self.params.get('resolution', self.d_resolution), 
                in_res=m_spec_i.resolution, 
                rebin=False, 
                gamma=self.params.get('gamma', None),
                fwhm=self.params.get('fwhm', None),
                )

            # Rebin onto the data's wavelength grid
            m_spec_i.rebin(d_wave=self.d_wave[i,:], replace_wave_flux=True)

            wave[i,:,:] = m_spec_i.wave
            flux[i,:,:] = m_spec_i.flux
            

        # Create a new ModelSpectrum instance with all orders
        m_spec = ModelSpectrum(
            wave=wave, 
            flux=flux, 
            lbl_opacity_sampling=self.lbl_opacity_sampling, 
            )
    
        # Save memory, same attributes in DataSpectrum
        del m_spec.wave, m_spec.mask_isfinite

        return m_spec



