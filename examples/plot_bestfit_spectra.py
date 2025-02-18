"""
Plot best-fit spectra and residuals for multiple orders.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pathlib

from red_dwarf_isotopes.utils import get_data_file, get_data_dir


def main(target: str, color: str):
    
    
    fig, ax = fig_ax()
    lw = 0.8
    
    file = get_data_file('bestfit_spec.npy', target=target)
    wave, flux, err, mask, m, spline_cont = np.load(file)

    # Divide best-fit spline continuum from flux
    flux /= spline_cont
    err /= spline_cont
    m /= spline_cont
    
    # Calculate residuals
    res = flux - m
    
    mask = mask.astype(bool)
    
    assert len(ax) == 9, 'ax must be a list of two Axes objects'
    ax_spec = [ax[i] for i in range(0, 9, 3)]
    ax_res = [ax[i] for i in range(1, 9, 3)]
    ax_hide = [ax[i] for i in range(2, 9, 3)]
    
    n_orders = wave.shape[0]
    for order in range(n_orders):
        ax_spec[order].plot(wave[order], flux[order], color='k', lw=lw)
        ax_spec[order].fill_between(wave[order], flux[order] - err[order], flux[order] + err[order], alpha=0.2, color='k', lw=0)
        ax_spec[order].plot(wave[order], m[order], color=color, lw=lw)
        
        ax_res[order].fill_between(wave[order], -err[order], err[order], alpha=0.4, color=color, lw=0)
        ax_res[order].plot(wave[order], res[order], label=f'Order {order}', color='k', alpha=0.8, 
                           marker='o', ms=1, ls='')
        ax_res[order].axhline(0, color='k', lw=lw*0.5)
        
        xlim = np.nanmin(wave[order]), np.nanmax(wave[order])
        ax_spec[order].set_xlim(xlim)
        ax_res[order].set_xlim(xlim)
        
        # hide x-axis on ax_spec
        ax_spec[order].set_xticks([])
        ax_spec[order].set(ylabel='Normalised flux')
        
        # show x-axis on ax_res
        ax_res[order].set_xlabel('Wavelength / nm')
        ax_res[order].set(ylabel='Residuals')
        
        # hide spines and ticks and plot
        ax_hide[order].remove()
        
        
    ax[0].set_title(get_title(target))

    # save figure as PDF
    fig.savefig(get_data_dir() / target / 'bestfit_spectra.pdf', bbox_inches='tight')
    print(f'Saved {get_data_dir() / target / "bestfit_spectra.pdf"}')
    
    return fig, ax
if __name__ == '__main__':
    
    file = get_data_file('fundamental_parameters.csv')
    
    interactive = False

    # Load data table with the fundamental parameters of the sample of stars
    # columns are: (Star, SpT, Distance (pc), M/M_sun, Teff (K), log g, [M/H])
    sample = pd.read_csv(file)
    
    names = sample['Star'].to_list()
    targets = [s.replace('Gl ', 'gl') for s in names]
    teff =  dict(zip(targets, [float(t.split('+-')[0]) for t in sample['Teff (K)'].to_list()]))
    
    # Create a color map for the effective temperature
    norm = plt.Normalize(3000, 3900.0)
    cmap = plt.cm.coolwarm_r

    # Create figure
    
    def fig_ax():
        fig, ax = plt.subplots(9, 1, figsize=(14, 9), 
                               gridspec_kw={'hspace': 0.10,
                                            'height_ratios': 3*[2,1,0.8]})
        return fig, ax
    
    def get_title(target):
        title = 'Gl ' + r'$\bf{' + target.replace('gl', '') + '}$'
        # add spacing
        title += '\n'
        title += r'SpT = ' + f'{sample["SpT"].iloc[t]}, '
        title += r'$T_{eff}$ = ' + f'{sample["Teff (K)"].iloc[t]} K, '
        title += r'[M/H] = ' + f'{sample["[M/H]"].iloc[t]}'
        return title
    
    for t, target in enumerate(targets):
        print(f'Plotting {t+1} of {len(targets)}: {target}')
        fig, ax = main(target, color=cmap(norm(teff[target])))
        plt.close()


