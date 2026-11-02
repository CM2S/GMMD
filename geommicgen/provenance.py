"""
Module saying which code a run is made with.

A run is the input file it is given and the code that carries it out: the commit of GMMD
and the versions of the libraries whose results it depends on. The microstructure file
records them beside the input file's name, so that a sample says what made it.
"""

import functools
import importlib.metadata
import os
import subprocess

LIBRARIES = ("numpy", "scipy", "gmsh")
# The libraries whose versions can change what a run gives, and are recorded with it


@functools.lru_cache(maxsize=None)
def code_provenance():
    """
    Give the code a microstructure is generated with.

    The commit is asked of git when a run starts, not when GMMD is installed, so that
    an editable install says the commit it runs, not the one it was installed at.

    Returns
    -------
    dict
        *git_commit*, the commit GMMD is checked out at, and *git_dirty*, whether a
        file of the package differs from it, both None outside a git checkout; and
        *libraries*, the version of each of `LIBRARIES`, None for one not installed.
    """
    package_dir = os.path.dirname(os.path.abspath(__file__))
    commit = dirty = None
    try:
        subprocess.run(
            ["git", "ls-files", "--error-unmatch", "__init__.py"],
            cwd=package_dir,
            capture_output=True,
            check=True,
        )
        # The package is a file of the checkout, not an install that lies inside one
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=package_dir,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain", "--", "."],
                cwd=package_dir,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        commit = dirty = None
    # A file of the package changed, staged or not yet tracked makes the run differ
    # from the commit; what lies outside the package, the results of runs among it,
    # does not

    libraries = {}
    for i_name in LIBRARIES:
        try:
            libraries[i_name] = importlib.metadata.version(i_name)
        except importlib.metadata.PackageNotFoundError:
            libraries[i_name] = None
    # Read from the installed packages, without importing them

    return {"git_commit": commit, "git_dirty": dirty, "libraries": libraries}
