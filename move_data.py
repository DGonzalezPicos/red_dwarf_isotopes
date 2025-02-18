import os
import pathlib
import shutil
import numpy as np

from retrieval_base.auxiliary_functions import spirou_sample, read_spirou_sample_csv
from retrieval_base.config import Config

# for t, name in enumerate(names):

def main(target, run=None):
    
    if target not in os.getcwd():
        os.chdir(base_path + target)
    # config_file = 'config_freechem.txt'
    # conf = Config(path=base_path, target=target, run=run)(config_file)
    outputs = pathlib.Path(base_path) / target / 'retrieval_outputs'
    # find dirs in outputs
    # print(f' outputs = {outputs}')
    dirs = [d for d in outputs.iterdir() if d.is_dir() and 'fc' in d.name and '_' not in d.name]
    # print(f' dirs = {dirs}')
    runs = [int(d.name.split('fc')[-1]) for d in dirs]
    # print(f' runs = {runs}')
    # print(f' {target}: Found {len(runs)} runs: {runs}')
    # assert len(runs) > 0, f'No runs found in {outputs}'
    if run is None:
        run = 'fc'+str(max(runs))
    else:
        run = 'fc'+str(run)
        assert run in [d.name for d in dirs], f'Run {run} not found in {dirs}'
    # print('Run:', run)
    # check that the folder 'test_output' is not empty
    test_output = outputs / run / 'test_output'
    assert test_output.exists(), f'No test_output folder found in {test_output}'
    
    bestfit_spec_file = test_output / 'bestfit_spec.npy'
    new_dir = pathlib.Path(new_path) / target
    new_dir.mkdir(parents=True, exist_ok=True)
    if bestfit_spec_file.exists():
        # assert bestfit_spec_file.exists(), f' Bestfit model not found in {bestfit_spec_file}'
            
        print(f' Bestfit model found in {bestfit_spec_file}')
        wave, flux, err, mask, m, spline_cont = np.load(bestfit_spec_file)
        print(f' Bestfit model loaded from {bestfit_spec_file}')
        mask = mask.astype(bool)

        # Remove the existing file if it exists and copy the new one
        target_file = new_dir / 'bestfit_spec.npy'
        if target_file.exists():
            print(f' Removing existing file {target_file}')
            target_file.unlink()  # Remove existing file
        print(f' Copying bestfit_spec.npy to {target_file}')
        # shutil.copy2(bestfit_spec_file, target_file)  # Copy with metadata preservation
        shutil.copy(bestfit_spec_file, target_file)
        
    else:
        print(f' Bestfit model not found in {bestfit_spec_file}')
    
    # now do the opposite: copy back the bestfit_spec.npy file to the retrieval_outputs folder
        shutil.copy(new_dir / 'bestfit_spec.npy', outputs / run / 'test_output' / 'bestfit_spec.npy')
        print(f' Bestfit model copied back to {outputs / run / "test_output" / "bestfit_spec.npy"}')
    
    

new_path = '/home/dario/phd/red_dwarf_isotopes/data/'
base_path = '/home/dario/phd/retrieval_base/'
df = read_spirou_sample_csv()
names = df['Star'].to_list()

targets = [s.replace('Gl ', 'gl') for s in names]

for t in targets:
    main(t)