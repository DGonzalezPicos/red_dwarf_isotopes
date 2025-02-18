"""Utility functions for path management and data access."""

import os
from pathlib import Path
import pkg_resources

def get_project_root() -> Path:
    """Return the absolute path to the project root directory.
    
    Returns
    -------
    Path
        Absolute path to the project root directory
    """
    return Path(__file__).parent.parent.absolute()

def get_data_dir() -> Path:
    """Return the absolute path to the data directory.
    
    Returns
    -------
    Path
        Absolute path to the data directory
    """
    return get_project_root() / 'data'

def get_data_file(filename: str, target: str = None) -> Path:
    """Get the absolute path to a data file.
    
    Parameters
    ----------
    filename : str
        Name of the file in the data directory, can include subdirectories
        e.g., 'gl15A/bestfit_spec.npy'
    
    Returns
    -------
    Path
        Absolute path to the data file
    
    Raises
    ------
    FileNotFoundError
        If the file does not exist in the data directory
    """
    if target is not None:
        file_dir = get_data_dir() / target
    else:
        file_dir = get_data_dir()
    file_path = file_dir / filename
    if not file_path.exists():
        raise FileNotFoundError(
            f"Data file '{filename}' not found in {file_dir}"
        )
    return file_path

def ensure_dir_exists(path: Path) -> None:
    """Ensure that a directory exists, creating it if necessary.
    
    Parameters
    ----------
    path : Path
        Directory path to ensure exists
    """
    path.mkdir(parents=True, exist_ok=True) 