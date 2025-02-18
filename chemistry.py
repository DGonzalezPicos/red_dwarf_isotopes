"""
Chemistry module for stellar atmosphere modeling.

This module provides classes for handling chemical composition and abundances
in stellar atmospheres, with particular focus on molecular species and isotopes
relevant for red dwarf stars.
"""

from typing import Dict, List, Optional, Tuple, Union, Type, Any
import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
import petitRADTRANS.nat_cst as nc
import os
import h5py

from red_dwarf_isotopes.utils import get_data_file

ArrayType = npt.NDArray[np.float64]

def get_chemistry_class(
    line_species: Union[List[str], Dict[str, str]],
    pressure: ArrayType,
    mode: str,
    **kwargs
) -> 'Chemistry':
    """
    Factory function to create appropriate chemistry class.
    
    Parameters
    ----------
    line_species : Union[List[str], Dict[str, str]]
        Species to include in calculations
    pressure : ArrayType
        Atmospheric pressure grid
    mode : str
        Type of chemistry ('fastchem' for equilibrium chemistry)
    **kwargs
        Additional arguments passed to chemistry class
        
    Returns
    -------
    Chemistry
        Instance of appropriate chemistry class
        
    Raises
    ------
    ValueError
        If mode is not recognized
    """
    chemistry_classes = {
        'fastchem': FastChemistry
    }
    
    if mode not in chemistry_classes:
        raise ValueError(f"Unknown chemistry mode: {mode}. Available modes: {list(chemistry_classes.keys())}")
    
    return chemistry_classes[mode](line_species, pressure, **kwargs)


class Chemistry:
    """
    Base class for chemical composition calculations.
    
    Attributes
    ----------
    species_info_default_file : str
        Path to default species information file
    line_species : List[str]
        Species included in calculations
    pressure : ArrayType
        Atmospheric pressure grid
    n_atm_layers : int
        Number of atmospheric layers
    mass_fractions_envelopes : Optional[Dict[str, ArrayType]]
        Mass fraction envelopes if available
    mass_fractions_posterior : Optional[Dict[str, ArrayType]]
        Posterior mass fractions if available
    """
    
    species_info_default_file = get_data_file('species_info.csv')
    
    def __init__(
        self,
        line_species: List[str],
        pressure: ArrayType
    ) -> None:
        """
        Initialize Chemistry object.
        
        Parameters
        ----------
        line_species : List[str]
            Species to include in calculations
        pressure : ArrayType
            Atmospheric pressure grid
        """
        self.line_species = line_species
        self.pressure = pressure
        self.n_atm_layers = len(self.pressure)
        
        self.mass_fractions_envelopes: Optional[Dict[str, ArrayType]] = None
        self.mass_fractions_posterior: Optional[Dict[str, ArrayType]] = None

    def set_species_info(
        self,
        line_species_dict: Optional[Dict[str, str]] = None,
        file: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Set up species information from file and/or dictionary.
        
        Parameters
        ----------
        line_species_dict : Optional[Dict[str, str]], optional
            Dictionary mapping species names to pRT names
        file : Optional[str], optional
            Path to species information file
            
        Returns
        -------
        pd.DataFrame
            Species information dataframe
            
        Raises
        ------
        AssertionError
            If species information is not loaded
        """
        if file is not None:
            self.species_info = pd.read_csv(file)
            self.species_info['label'] = self.species_info['mathtext_name']
            
        if not hasattr(self, 'species_info'):
            raise AssertionError('species_info not yet loaded')
            
        if line_species_dict is not None:
            line_species_dict_default = dict(zip(
                self.species_info['name'].tolist(),
                self.species_info['pRT_name'].tolist()
            ))
            line_species_dict_new = line_species_dict_default.copy()
            line_species_dict_new.update(line_species_dict)
            
            self.species_info['pRT_name'] = self.species_info['name'].map(line_species_dict_new)
        
        # Create forward and reverse name mappings
        self.pRT_name_dict = {
            v['pRT_name']: v['name'] 
            for _, v in self.species_info.iterrows()
        }
        self.pRT_name_dict_r = {
            v['name']: v['pRT_name'] 
            for _, v in self.species_info.iterrows()
        }
        
        return self.species_info

    def read_species_info(
        self,
        species: str,
        info_key: str
    ) -> Any:
        """
        Read specific information for a species.
        
        Parameters
        ----------
        species : str
            Species name
        info_key : str
            Information key to retrieve
            
        Returns
        -------
        Any
            Requested information
            
        Raises
        ------
        AssertionError
            If species or key not found
        """
        if species not in self.species_info['name'].values:
            raise AssertionError(f'species = {species} not in species_info')
        if info_key not in self.species_info.columns:
            raise AssertionError(f'info_key = {info_key} not in species_info.columns')
            
        return self.species_info.loc[
            self.species_info['name'] == species,
            info_key
        ].values[0]

    def get_VMRs_posterior(
        self,
        profile_id: int = 0,
        save_to: Optional[str] = None
    ) -> 'Chemistry':
        """
        Calculate volume mixing ratio posteriors and envelopes.
        
        Parameters
        ----------
        profile_id : int, optional
            Profile index to use, by default 0
        save_to : Optional[str], optional
            Path to save results, by default None
            
        Returns
        -------
        Chemistry
            Self with updated VMRs
            
        Raises
        ------
        AssertionError
            If mass fractions posterior not calculated
        """
        if self.mass_fractions_posterior is None:
            raise AssertionError('mass_fractions_posterior not yet calculated')
            
        self.VMRs_posterior = {}
        self.VMRs_envelopes = {}
        MMW = self.mass_fractions['MMW']
        
        # Calculate VMRs for each species
        for line_species_i in self.line_species:
            key_i = self.pRT_name_dict.get(line_species_i)
            if not key_i:
                continue
                
            mu = self.read_species_info(key_i, 'mass')
            vmr_i = self.mass_fractions_posterior[line_species_i] * (MMW / mu)
            self.VMRs_posterior[key_i] = vmr_i[:, profile_id]
            self.VMRs_envelopes[key_i] = np.quantile(vmr_i, [0.16, 0.5, 0.84], axis=0)
        
        # Calculate isotope ratios
        self._calculate_isotope_ratios()
        
        # Save results if requested
        if save_to is not None:
            self._save_VMRs(save_to)
            
        return self

    def _calculate_isotope_ratios(self) -> None:
        """Calculate isotope ratios from VMRs."""
        isotope_pairs = [
            ("13CO", "12CO", "12_13CO"),
            ("C18O", "12CO", "C16_18O"),
            ("H2O_181", "H2O", "H2_16_18O")
        ]
        
        for iso1, iso2, ratio_name in isotope_pairs:
            if iso1 in self.VMRs_posterior and iso2 in self.VMRs_posterior:
                if ratio_name == "C16_18O":
                    # Special handling for C18O detection
                    q16, q50, q84 = np.quantile(
                        self.VMRs_posterior["C18O"],
                        [0.16, 0.5, 0.84]
                    )
                    if all(abs(x - y) < 1.0 for x, y in [
                        (q84, q16), (q50, q16), (q84, q50)
                    ]):
                        self.VMRs_posterior[ratio_name] = (
                            self.VMRs_posterior[iso2] /
                            self.VMRs_posterior[iso1]
                        )
                else:
                    self.VMRs_posterior[ratio_name] = (
                        self.VMRs_posterior[iso2] /
                        self.VMRs_posterior[iso1]
                    )

    def _save_VMRs(self, save_to: str) -> None:
        """
        Save VMR results to files.
        
        Parameters
        ----------
        save_to : str
            Base path for saving files
        """
        files = {
            'posterior': save_to + 'posterior.npy',
            'envelopes': save_to + 'envelopes.npy',
            'labels': save_to + 'labels.npy'
        }
        
        np.save(files['posterior'], np.array(list(self.VMRs_posterior.values())))
        np.save(files['envelopes'], np.array(list(self.VMRs_envelopes.values())))
        np.save(files['labels'], np.array(list(self.VMRs_posterior.keys())))
        
        print(f'Saved VMRs to {save_to}*')


class FastChemistry(Chemistry):
    """
    FastChem equilibrium chemistry calculations.
    
    This class handles chemical equilibrium calculations using the FastChem
    grid approach, with special handling for isotopologues.
    
    Attributes
    ----------
    isotopologues_dict : Dict[str, List[str]]
        Mapping of main species to isotopologues
    isotopologues : List[str]
        List of all isotopologues
    """
    
    isotopologues_dict = {
        '12CO': ['13CO', 'C18O', 'C17O'],
        'H2O': ['H2O_181', 'H2O_171']
    }
    isotopologues_dict_rev = {
        value: key 
        for key, values in isotopologues_dict.items() 
        for value in values
    }
    isotopologues = list(isotopologues_dict_rev.keys())
    
    def __init__(
        self,
        line_species: Union[List[str], Dict[str, str]],
        pressure: ArrayType,
        **kwargs
    ) -> None:
        """
        Initialize FastChemistry object.
        
        Parameters
        ----------
        line_species : Union[List[str], Dict[str, str]]
            Species to include in calculations
        pressure : ArrayType
            Atmospheric pressure grid
        **kwargs
            Additional arguments including fastchem_grid_file
            
        Raises
        ------
        AssertionError
            If fastchem_grid_file not provided or not found
        """
        self.species_info = self.set_species_info(file=self.species_info_default_file)
        
        if isinstance(line_species, dict):
            self.species_info = self.set_species_info(line_species_dict=line_species)
            line_species = list(line_species.values())
            
        super().__init__(line_species, pressure)
        
        self.species = [
            self.pRT_name_dict.get(line_species_i)
            for line_species_i in self.line_species
        ]
        
        fc_grid_file = kwargs.get('fastchem_grid_file')
        if not fc_grid_file:
            raise AssertionError('No fastchem grid file given')
        if not os.path.exists(fc_grid_file):
            raise AssertionError(f'File {fc_grid_file} not found')
            
        self.load_grid(fc_grid_file)

    def load_grid(self, file: str) -> 'FastChemistry':
        """
        Load FastChem grid from HDF5 file.
        
        Parameters
        ----------
        file : str
            Path to FastChem grid file
            
        Returns
        -------
        FastChemistry
            Self with loaded grid
        """
        with h5py.File(file, 'r') as f:
            data = f['data'][:]
            self.t_grid = f.attrs['temperature']
            self.p_grid = f.attrs['pressure']
            columns = [
                col.decode() if isinstance(col, bytes) else col 
                for col in f.attrs['columns']
            ]
            species = f.attrs['species'].tolist()
            labels = f.attrs['labels'].tolist()
        
        self.data = np.moveaxis(data, 0, 1)
        self.fc_species_dict = dict(zip(species, labels))
        
        # Create interpolators for each species
        self.interpolator = {}
        for species_i, label_i in self.fc_species_dict.items():
            if (species_i in self.species) or (species_i in ['H2', 'He', 'e-']):
                if label_i in columns:
                    idx = columns.index(label_i)
                    self.interpolator[species_i] = RegularGridInterpolator(
                        (self.t_grid, self.p_grid),
                        self.data[:, :, idx],
                        bounds_error=False,
                        fill_value=None
                    )
        
        return self

    def __call__(
        self,
        params: Dict[str, float],
        temperature: ArrayType
    ) -> Dict[str, ArrayType]:
        """
        Calculate mass fractions for given parameters and temperature profile.
        
        Parameters
        ----------
        params : Dict[str, float]
            Chemistry parameters including abundances and scaling factors
        temperature : ArrayType
            Temperature profile
            
        Returns
        -------
        Dict[str, ArrayType]
            Mass fractions for each species
            
        Raises
        ------
        AssertionError
            If temperature profile length doesn't match pressure grid
        """
        if len(temperature) != len(self.pressure):
            raise AssertionError(
                f'Len(temperature) = {len(temperature)} != '
                f'len(pressure) = {len(self.pressure)}'
            )
            
        # Calculate volume mixing ratios
        self.VMRs = {
            k: self.interpolator[k]((temperature, self.pressure))
            for k in self.interpolator
        }
        
        # Initialize mass fractions
        self.mass_fractions = {}
        
        # Calculate mass fractions for each species
        for line_species_i, species_i in zip(self.line_species, self.species):
            if species_i is None:
                continue
                
            alpha_i = params.get(f'alpha_{species_i}', 0.0)
            mass_i = self.read_species_info(species_i, 'mass')
            
            if species_i in self.isotopologues:
                main = self.isotopologues_dict_rev[species_i]
                alpha_main = params.get(f'alpha_{main}', 0.0)
                ratio = params.get(f'{main}/{species_i}')
                
                if ratio is None:
                    raise AssertionError(f'No ratio {main}/{species_i} given')
                    
                self.VMRs[species_i] = (
                    self.VMRs[main] * 10.**alpha_main
                ) / ratio
                
            elif species_i in params:
                self.VMRs[species_i] = params[species_i] * np.ones(self.n_atm_layers)
                
            elif species_i not in self.VMRs:
                self.VMRs[species_i] = np.zeros(self.n_atm_layers)
                
            self.mass_fractions[line_species_i] = (
                mass_i * (self.VMRs[species_i] * 10.**alpha_i)
            )
        
        # Add background species
        self.mass_fractions.update({
            'He': self.read_species_info('He', 'mass') * self.VMRs['He'],
            'H2': self.read_species_info('H2', 'mass') * self.VMRs['H2'],
            'H': self.VMRs['H2'],
            'e-': 5.48579909070e-4 * self.VMRs['e-'],
            'H-': self.VMRs['e-']
        })
        
        # Calculate mean molecular weight
        MMW = np.sum(list(self.mass_fractions.values()), axis=0)
        
        # Normalize mass fractions
        for key in self.mass_fractions:
            self.mass_fractions[key] /= MMW
            
        self.mass_fractions['MMW'] = MMW
        
        return self.mass_fractions

    @property
    def CO(self) -> float:
        """C/O ratio."""
        return getattr(self, 'C_O', np.nan)
    
    @property
    def FeH(self) -> float:
        """[Fe/H] metallicity."""
        return getattr(self, 'Z', np.nan)

if __name__ == '__main__':
    
    kwargs = dict(fastchem_grid_file=get_data_file('fastchem_grid.h5'))
    
    pressure = np.logspace(-5, 2, 40)
    temperature = np.linspace(1000.0, 3000.0, len(pressure))
    opacity_params = {
    'log_H2O': ([(-12,-2), r'$\log\ \mathrm{H_2O}$'], 'H2O_pokazatel_main_iso'),
    'log_OH':  ([(-12,-2), r'$\log\ \mathrm{OH}$'], 'OH_MYTHOS_main_iso'),
    }
    
    keys = [k.split('log_')[-1] for k in opacity_params.keys()]
    values = [v[1] for v in opacity_params.values()]
    line_species_dict = dict(zip(keys, values))
    
    chem = FastChemistry(line_species=line_species_dict, pressure=pressure, **kwargs)

    
    params = {'alpha_H2O':-0.2,
              'alpha_OH':-0.2,
              }
    mf = chem(params, temperature)
    
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1,2, figsize=(12,6), sharey=True, tight_layout=True)
    ax[0].plot(temperature, pressure, 'k-')
    ax[0].set(yscale='log', xscale='linear', ylim=(pressure.max(), pressure.min()), xlabel='Temperature [K]', ylabel='Pressure [bar]')
    
    for i, (k,v) in enumerate(chem.VMRs.items()):
        ax[1].plot(v, pressure, label=k)
        
    ax[1].set(yscale='log', xscale='log', ylim=(pressure.max(), pressure.min()), xlabel='Mass fraction', ylabel='Pressure [bar]')
    ax[1].legend()
    plt.show()