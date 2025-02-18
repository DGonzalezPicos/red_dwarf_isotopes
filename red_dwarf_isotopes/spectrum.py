"""
Spectrum handling module.

"""

from typing import Optional, Union, Tuple, ClassVar
import numpy as np
import numpy.typing as npt
from scipy.ndimage import gaussian_filter1d, gaussian_filter, generic_filter
from scipy.interpolate import interp1d
from scipy.sparse import triu

import pickle
import os
import warnings
import matplotlib.pyplot as plt

from PyAstronomy import pyasl
import petitRADTRANS.nat_cst as nc
from spectres import spectres, spectres_numba

import retrieval_base.auxiliary_functions as af
import retrieval_base.figures as figs
from retrieval_base.spline_model import SplineModel

try:
    from broadpy import InstrumentalBroadening
    HAS_BROADPY = True
except ImportError:
    HAS_BROADPY = False

ArrayType = npt.NDArray[np.float64]

class Spectrum:
    """Base class for handling astronomical spectra.
    
    This class provides fundamental operations for spectral analysis including
    wavelength shifts, broadening, and normalization.
    
    Attributes
    ----------
    n_pixels : int
        Number of pixels in the spectrum
    wave : ArrayType
        Wavelength array
    flux : ArrayType
        Flux array
    err : Optional[ArrayType]
        Error array for flux measurements
    w_set : str
        Wavelength set identifier
    n_orders : int
        Number of spectral orders
    n_dets : int
        Number of detectors
    mask_isfinite : ArrayType
        Boolean mask for finite values
    n_data_points : int
        Number of valid data points
    """
    
    n_pixels: ClassVar[int] = 4096
    reshaped: ClassVar[bool] = False
    normalized: ClassVar[bool] = False
    flux_unit: ClassVar[str] = ''  # default: normalized flux [unitless]

    def __init__(self, 
                 wave: ArrayType, 
                 flux: ArrayType, 
                 err: Optional[ArrayType] = None, 
                 w_set: str = 'spirou') -> None:
        """Initialize Spectrum object.
        
        Parameters
        ----------
        wave : ArrayType
            Wavelength array
        flux : ArrayType
            Flux array
        err : Optional[ArrayType], optional
            Error array, by default None
        w_set : str, optional
            Wavelength set identifier, by default 'spirou'
        """
        self.wave = wave
        self.flux = flux
        self.err = err
        self.w_set = w_set
        
        self.n_orders, _ = self.flux.shape
        self.n_dets = 1

        self.update_isfinite_mask()

    def update_isfinite_mask(self, 
                            array: Optional[ArrayType] = None, 
                            check_err: bool = False) -> None:
        """Update the mask for finite values in the spectrum.
        
        Parameters
        ----------
        array : Optional[ArrayType], optional
            Array to check for finite values, by default None
        check_err : bool, optional
            Whether to check error array as well, by default False
        """
        if array is None:
            self.mask_isfinite = np.isfinite(self.flux)
        else:
            self.mask_isfinite = np.isfinite(array)
            
        if check_err and self.err is not None:
            self.mask_isfinite &= np.isfinite(self.err)
            
        self.n_data_points = np.sum(self.mask_isfinite)

    def rv_shift(self, 
                 rv: float, 
                 wave: Optional[ArrayType] = None, 
                 replace_wave: bool = False) -> ArrayType:
        """Apply radial velocity shift to wavelength array.
        
        Parameters
        ----------
        rv : float
            Radial velocity in km/s
        wave : Optional[ArrayType], optional
            Input wavelength array, by default None
        replace_wave : bool, optional
            Whether to replace internal wavelength array, by default False
            
        Returns
        -------
        ArrayType
            Shifted wavelength array
        """
        if wave is None:
            wave = np.copy(self.wave)

        wave_shifted = wave * (1 + rv/(nc.c*1e-5))
        if replace_wave:
            self.wave = wave_shifted
        
        return wave_shifted
    
    def rot_broadening(self, 
                      vsini: float, 
                      epsilon_limb: float = 0, 
                      wave: Optional[ArrayType] = None, 
                      flux: Optional[ArrayType] = None, 
                      replace_wave_flux: bool = False) -> Union[ArrayType, Tuple[ArrayType, ArrayType]]:
        """Apply rotational broadening to spectrum.
        
        Parameters
        ----------
        vsini : float
            Projected rotational velocity in km/s
        epsilon_limb : float, optional
            Limb darkening coefficient, by default 0
        wave : Optional[ArrayType], optional
            Input wavelength array, by default None
        flux : Optional[ArrayType], optional
            Input flux array, by default None
        replace_wave_flux : bool, optional
            Whether to replace internal arrays, by default False
            
        Returns
        -------
        Union[ArrayType, Tuple[ArrayType, ArrayType]]
            Broadened spectrum (flux or wave,flux)
        """
        if wave is None:
            wave = self.wave
        if flux is None:
            flux = self.flux

        wave_even = np.linspace(wave.min(), wave.max(), wave.size)
        flux_even = np.interp(wave_even, xp=wave, fp=flux)
        
        if vsini > 1.0:
            flux_rot_broad = pyasl.fastRotBroad(wave_even, flux_even, 
                                              epsilon=epsilon_limb, 
                                              vsini=vsini)
        else:
            flux_rot_broad = flux_even

        if replace_wave_flux:
            self.wave = wave_even
            self.flux = flux_rot_broad
            return flux_rot_broad
        
        return wave_even, flux_rot_broad

    @classmethod
    def instr_broadening(cls, 
                        wave: ArrayType, 
                        flux: ArrayType, 
                        out_res: float = 1e6, 
                        in_res: float = 1e6) -> ArrayType:
        """Apply instrumental broadening using a Gaussian profile.
        
        Parameters
        ----------
        wave : ArrayType
            Wavelength array
        flux : ArrayType
            Flux array
        out_res : float, optional
            Output resolution, by default 1e6
        in_res : float, optional
            Input resolution, by default 1e6
            
        Returns
        -------
        ArrayType
            Broadened flux array
        """
        sigma_LSF = np.sqrt(1/out_res**2 - 1/in_res**2) / (2*np.sqrt(2*np.log(2)))
        spacing = np.mean(2*np.diff(wave) / (wave[1:] + wave[:-1]))
        sigma_LSF_gauss_filter = sigma_LSF / spacing
        
        return gaussian_filter(flux, sigma=sigma_LSF_gauss_filter, mode='nearest')
    
    @classmethod
    def instr_broadening_voigt(cls, 
                              wave: ArrayType, 
                              flux: ArrayType, 
                              fwhm: float, 
                              gamma: float = 0.0) -> ArrayType:
        """Apply instrumental broadening using a Voigt profile.
        
        Parameters
        ----------
        wave : ArrayType
            Wavelength array
        flux : ArrayType
            Flux array
        fwhm : float
            Full width at half maximum
        gamma : float, optional
            Lorentzian component, by default 0.0
            
        Returns
        -------
        ArrayType
            Broadened flux array
            
        Raises
        ------
        ImportError
            If broadpy package is not installed
        """
        if not HAS_BROADPY:
            raise ImportError("broadpy package required for Voigt profile broadening")
        return InstrumentalBroadening(wave, flux)(fwhm=fwhm, gamma=gamma)
    
    @classmethod
    def spectrally_weighted_integration(cls, wave: ArrayType, flux: ArrayType, array: ArrayType) -> float:
        """Integrate and weigh the array by the spectrum.
        
        Parameters
        ----------
        wave : ArrayType
            Wavelength array
        flux : ArrayType
            Flux array
        array : ArrayType
            Array to integrate and weigh
            
        Returns
        -------
        float
            Integrated and weighed array
        """
        integral1 = np.trapz(wave*flux*array, wave)
        integral2 = np.trapz(wave*flux, wave)

        return integral1/integral2
    
    def normalize_flux_per_order(self, 
                               fun: str = 'median', 
                               tell_threshold: float = 0.30) -> 'Spectrum':
        """Normalize flux in each spectral order.
        
        Parameters
        ----------
        fun : str, optional
            Function to use for normalization ('median' or 'mean'), by default 'median'
        tell_threshold : float, optional
            Threshold for telluric line masking, by default 0.30
            
        Returns
        -------
        Spectrum
            Self with normalized flux
        """
        deep_lines = (self.transm < tell_threshold 
                     if hasattr(self, 'transm') 
                     else np.zeros_like(self.flux, dtype=bool))
        
        f = np.where(deep_lines, np.nan, self.flux)
        self.norm = getattr(np, f'nan{fun}')(f, axis=-1)
        value = self.norm[..., None]
        self.flux /= value
        
        if self.err is not None:
            self.err /= value
            
        self.normalized = True
        return self
    
    def fill_nans(self, min_finite_pixels=100, debug=True):
        '''Fill NaNs order-detector pairs with less than `min_finite_pixels` finite pixels'''
        assert self.reshaped, 'The spectrum has not been reshaped yet!'
        
        for order in range(self.n_orders):
            for det in range(self.n_dets):
                mask_ij = self.mask_isfinite[order,det]
                if mask_ij.sum() < min_finite_pixels:
                    if debug:
                        print(f'[fill_nans] Order {order}, detector {det} has only {mask_ij.sum()} finite pixels! Setting all to NaN.')
                    self.flux[order,det,:] = np.nan * np.ones_like(self.flux[order,det,:])
        self.update_isfinite_mask()
        return self
    

class ModelSpectrum(Spectrum):
    """Class for handling model spectra with additional functionality.
    
    This class extends the base Spectrum class with additional features
    specific to model spectra, such as rebinning and combined broadening operations.
    
    Attributes
    ----------
    resolution : int
        Spectral resolution of the model
    """

    def __init__(self, 
                 wave: ArrayType, 
                 flux: ArrayType, 
                 lbl_opacity_sampling: int = 1) -> None:
        """Initialize ModelSpectrum object.
        
        Parameters
        ----------
        wave : ArrayType
            Wavelength array
        flux : ArrayType
            Flux array
        lbl_opacity_sampling : int, optional
            Line-by-line opacity sampling factor, by default 1
        """
        super().__init__(wave, flux)

        self.n_orders, self.n_dets, self.n_pixels = self.flux.shape

        # Update the order wavelength ranges
        mask_order_wlen_ranges = (
            (self.order_wlen_ranges.min(axis=(1,2)) > self.wave.min() - 5) & 
            (self.order_wlen_ranges.max(axis=(1,2)) < self.wave.max() + 5)
        )
        self.order_wlen_ranges = self.order_wlen_ranges[mask_order_wlen_ranges,:,:]

        # Model resolution depends on the opacity sampling
        self.resolution = int(1e6/lbl_opacity_sampling)

    def rebin(self, 
              d_wave: ArrayType, 
              kind: str = 'spectres', 
              replace_wave_flux: bool = True) -> Optional[ArrayType]:
        """Rebin spectrum to new wavelength grid.
        
        Parameters
        ----------
        d_wave : ArrayType
            Target wavelength grid
        kind : str, optional
            Rebinning method ('spectres' or 'linear'), by default 'spectres'
        replace_wave_flux : bool, optional
            Whether to replace internal arrays, by default True
            
        Returns
        -------
        Optional[ArrayType]
            Rebinned flux array if replace_wave_flux is False
            
        Notes
        -----
        The 'spectres' method is recommended for accuracy.
        """
        d_wave = np.atleast_2d(d_wave)
        
        if kind == 'spectres':
            rebinned = []
            for wave_i in d_wave:
                rebinned.append(
                    spectres_numba(wave_i, self.wave, self.flux, fill=np.nan)
                )
            new_flux = np.array(rebinned)
        elif kind == 'linear':
            new_flux = np.interp(d_wave, self.wave, self.flux)
        else:
            raise ValueError(f"Unknown rebinning method: {kind}")
            
        if replace_wave_flux:
            self.wave = d_wave
            self.flux = new_flux
            return None
            
        return new_flux

    def shift_broaden_rebin(self, 
                           rv: float, 
                           vsini: float, 
                           d_wave: Optional[ArrayType] = None, 
                           epsilon_limb: float = 0, 
                           out_res: float = 1e6, 
                           in_res: float = 1e6, 
                           rebin: bool = True, 
                           gamma: Optional[float] = None,
                           fwhm: Optional[float] = None) -> None:
        """Apply combined wavelength shift, broadening, and rebinning.
        
        This method combines multiple spectral operations in the correct order:
        1. Radial velocity shift
        2. Rotational broadening
        3. Instrumental broadening (Gaussian or Voigt)
        4. Rebinning to target wavelength grid
        
        Parameters
        ----------
        rv : float
            Radial velocity in km/s
        vsini : float
            Projected rotational velocity in km/s
        d_wave : Optional[ArrayType], optional
            Target wavelength grid for rebinning, by default None
        epsilon_limb : float, optional
            Limb darkening coefficient, by default 0
        out_res : float, optional
            Output resolution, by default 1e6
        in_res : float, optional
            Input resolution, by default 1e6
        rebin : bool, optional
            Whether to perform rebinning, by default True
        gamma : Optional[float], optional
            Lorentzian component for Voigt profile, by default None
        fwhm : Optional[float], optional
            FWHM for Voigt profile, by default None
            
        Raises
        ------
        ValueError
            If gamma is provided without fwhm for Voigt profile
        """
        # Apply operations in sequence
        self.rv_shift(rv, replace_wave=True)
        self.rot_broadening(vsini, epsilon_limb, replace_wave_flux=True)
        
        if gamma is not None:
            if fwhm is None:
                raise ValueError("FWHM must be provided when using Voigt profile")
            self.flux = self.instr_broadening_voigt(self.wave, self.flux, fwhm, gamma)
        else:
            self.flux = self.instr_broadening(self.wave, self.flux, out_res, in_res)
            
        if rebin and d_wave is not None:
            self.rebin(d_wave, replace_wave_flux=True)
