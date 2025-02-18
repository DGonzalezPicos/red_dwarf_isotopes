"""
Pressure-Temperature profile handling for stellar atmospheres.

This module provides classes for handling different types of pressure-temperature
profiles, particularly for radiative-convective equilibrium atmospheres.
"""

from typing import Dict, List, Optional, Tuple, Union, Type
import numpy as np
import numpy.typing as npt
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

ArrayType = npt.NDArray[np.float64]

def get_PT_profile_class(
    pressure: ArrayType,
    mode: str,
    **kwargs
) -> 'PTProfile':
    """
    Factory function to create appropriate PT profile class.
    
    Parameters
    ----------
    pressure : ArrayType
        Atmospheric pressure grid
    mode : str
        Type of PT profile ('RCE' for radiative-convective equilibrium)
    **kwargs
        Additional arguments passed to profile class
        
    Returns
    -------
    PTProfile
        Instance of appropriate PT profile class
        
    Raises
    ------
    ValueError
        If mode is not recognized
    """
    profile_classes = {
        'RCE': PTProfileRCE
    }
    
    if mode not in profile_classes:
        raise ValueError(f"Unknown PT profile mode: {mode}. Available modes: {list(profile_classes.keys())}")
    
    return profile_classes[mode](pressure, **kwargs)


class PTProfile:
    """
    Base class for pressure-temperature profiles.
    
    Attributes
    ----------
    pressure : ArrayType
        Atmospheric pressure grid
    temperature_envelopes : Optional[ArrayType]
        Temperature range envelopes if available
    int_contr_em : Dict
        Integrated emission contributions
    """
    
    def __init__(self, pressure: ArrayType) -> None:
        """
        Initialize PT profile.
        
        Parameters
        ----------
        pressure : ArrayType
            Atmospheric pressure grid
        """
        self.pressure = pressure
        self.temperature_envelopes: Optional[ArrayType] = None
        self.int_contr_em: Dict = {}


class PTProfileRCE(PTProfile):
    """
    Temperature profile for a radiative-convective equilibrium atmosphere.
    
    This class handles atmospheres with a single convective region, computing
    the temperature profile based on the pressure structure and gradient parameters.
    
    Attributes
    ----------
    PT_interp_mode : str
        Interpolation mode for PT profile
    log10_pressure : ArrayType
        Log10 of pressure grid
    fix_bounds : bool
        Whether to fix profile boundaries
    PT_adiabatic : bool
        Whether to use adiabatic profile
    """
    
    def __init__(
        self,
        pressure: ArrayType,
        PT_interp_mode: str = 'quadratic',
        **kwargs
    ) -> None:
        """
        Initialize RCE profile.
        
        Parameters
        ----------
        pressure : ArrayType
            Atmospheric pressure grid
        PT_interp_mode : str, optional
            Interpolation mode for PT profile, by default 'quadratic'
        **kwargs
            Additional parameters to set as attributes
        """
        super().__init__(pressure)
        
        self.flipped_ln_pressure = np.log(self.pressure)[::-1]
        self.PT_interp_mode = PT_interp_mode
        self.log10_pressure = np.log10(self.pressure)
        
        # Set default attributes
        self.fix_bounds = True
        self.PT_adiabatic = True
        
        # Update with any provided kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __call__(self, params: Dict[str, Union[float, ArrayType]]) -> ArrayType:
        """
        Compute temperature profile for given parameters.
        
        Parameters
        ----------
        params : Dict[str, Union[float, ArrayType]]
            Parameters for profile calculation including:
            - log_P_RCE: Log pressure at RCE point
            - dlnT_dlnP_knots: Temperature gradient at knot points
            - T_0: Surface temperature
            - dlog_P_1, dlog_P_3: Pressure spacing (optional)
            
        Returns
        -------
        ArrayType
            Temperature profile
            
        Raises
        ------
        AssertionError
            If required parameters are missing
        """
        required_params = ['log_P_RCE', 'dlnT_dlnP_knots', 'T_0']
        for param in required_params:
            if param not in params:
                raise AssertionError(f"Missing required parameter: {param}")
        
        # Set up pressure knots
        n_knots = len(params['dlnT_dlnP_knots'])
        self.log_P_knots = np.ones(n_knots)
        self.log_P_knots[0] = self.log10_pressure.max()
        self.log_P_knots[-1] = self.log10_pressure.min()
        
        # Handle pressure spacing
        if 'dlog_P_1' in params:
            dlog_P = [params['dlog_P_1'], params['dlog_P_3']]
        else:
            dlog_P = [params['dlog_P'], params['dlog_P']]
        
        # Set up additional knots for 7-point profile
        x_factor = 2.0 if n_knots == 7 else 1.0
        if n_knots == 7:
            self.log_P_knots[2] = min(params['log_P_RCE'] + dlog_P[0], self.log_P_knots[0]*0.8)
            self.log_P_knots[4] = max(params['log_P_RCE'] - dlog_P[1], self.log_P_knots[-1]*0.8)
            
        # Set intermediate points
        self.log_P_knots[1] = min(params['log_P_RCE'] + x_factor*dlog_P[0], self.log10_pressure.max()*0.9)
        self.log_P_knots[-2] = max(params['log_P_RCE'] - x_factor*dlog_P[1], self.log10_pressure.min()*0.9)
        
        # Set RCE point
        self.log_P_knots[n_knots//2] = params['log_P_RCE']
        
        # Interpolate temperature gradient
        interp_func = interp1d(
            self.log_P_knots[::-1],
            params['dlnT_dlnP_knots'][::-1],
            kind=self.PT_interp_mode
        )
        dlnT_dlnP_array = interp_func(self.log10_pressure)[::-1]
        
        # Ensure non-negative gradient
        dlnT_dlnP_array = np.maximum(dlnT_dlnP_array, 0.0)
        
        # Compute temperature profile
        self.temperature = self._compute_temperature(
            params['T_0'],
            dlnT_dlnP_array,
            self.flipped_ln_pressure
        )
        self.dlnT_dlnP_array = dlnT_dlnP_array[::-1]
        
        return self.temperature
    
    @staticmethod
    def _compute_temperature(
        T_0: float,
        gradients: ArrayType,
        ln_pressure: ArrayType
    ) -> ArrayType:
        """
        Compute temperature profile from gradients.
        
        Parameters
        ----------
        T_0 : float
            Surface temperature
        gradients : ArrayType
            Temperature gradients
        ln_pressure : ArrayType
            Log pressure grid
            
        Returns
        -------
        ArrayType
            Temperature profile
        """
        temperatures = [T_0]
        for i in range(len(ln_pressure)-1):
            T_next = temperatures[-1] * np.exp(
                (ln_pressure[i+1] - ln_pressure[i]) * gradients[i]
            )
            temperatures.append(T_next)
        return np.array(temperatures)[::-1]

    def plot(
        self,
        ax: Optional[plt.Axes] = None,
        **kwargs
    ) -> plt.Axes:
        """
        Plot temperature profile.
        
        Parameters
        ----------
        ax : Optional[plt.Axes], optional
            Axes to plot on, by default None
        **kwargs
            Additional arguments passed to plot
            
        Returns
        -------
        plt.Axes
            Axes object with plot
        """
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(6, 4))
        
        kwargs.setdefault('lw', 2)
        kwargs.setdefault('color', 'brown')
        
        ax.plot(self.temperature, self.pressure, **kwargs)
        ax.set_ylim(self.pressure.max(), self.pressure.min())
        ax.set(yscale='log', xlabel='Temperature [K]', ylabel='Pressure [bar]')
        
        return ax
    
    def plot_gradient(
        self,
        ax: Optional[plt.Axes] = None,
        **kwargs
    ) -> plt.Axes:
        """
        Plot temperature gradient profile.
        
        Parameters
        ----------
        ax : Optional[plt.Axes], optional
            Axes to plot on, by default None
        **kwargs
            Additional arguments passed to plot
            
        Returns
        -------
        plt.Axes
            Axes object with plot
        """
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(6, 4))
        
        kwargs.setdefault('lw', 2)
        kwargs.setdefault('color', 'darkgreen')
        
        if hasattr(self, 'dlnT_dlnP_envelopes'):
            ax.fill_betweenx(
                self.pressure,
                self.dlnT_dlnP_envelopes[0],
                self.dlnT_dlnP_envelopes[-1],
                alpha=0.1
            )
        else:
            ax.plot(self.dlnT_dlnP_array, self.pressure, **kwargs)
        
        # Plot pressure levels
        for log_P in self.log_P_knots:
            ax.axhline(10**log_P, color='k', linestyle='--', linewidth=0.5)
            
        ax.set_ylim(self.pressure.max(), self.pressure.min())
        ax.set(yscale='log', xlabel='dlnT/dlnP', ylabel='Pressure [bar]')
        
        return ax