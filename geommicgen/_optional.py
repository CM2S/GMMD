"""
Module containing the handling of the optional dependencies.

Some features of geommicgen rely on packages that are not required to generate a
microstructure, and which are therefore not installed by default. This module gives
those features a single place to ask for such a package, so that a missing dependency
is reported as a sentence naming the command that installs it, rather than as an
import error raised deep inside the call stack.
"""

# pylint: disable=import-error
# pylint: disable=relative-beyond-top-level
from geommicgen.errors.error_classes import MissingOptionalDependency

MINIMUM_GMSH_VERSION = (4, 15)
# Earliest API version of gmsh that matches the calls used here

GMSH_INSTALL_COMMAND = "pip install 'geommicgen[gmsh]'"


def require_gmsh():
    """
    Get the gmsh module, checking that it is installed and recent enough.

    Returns
    -------
    module
        The gmsh module.

    Raises
    ------
    MissingOptionalDependency:
        If gmsh is not installed, or if the installed version is older than
        `MINIMUM_GMSH_VERSION`.
    """
    try:
        import gmsh
    except ImportError:
        raise MissingOptionalDependency("gmsh", GMSH_INSTALL_COMMAND) from None

    needed = ".".join(str(i_part) for i_part in MINIMUM_GMSH_VERSION)
    try:
        version = (gmsh.GMSH_API_VERSION_MAJOR, gmsh.GMSH_API_VERSION_MINOR)
    except AttributeError:
        raise MissingOptionalDependency(
            "gmsh",
            GMSH_INSTALL_COMMAND,
            reason="the installed version is too old to report its API version and "
            "at least {0} is needed".format(needed),
        ) from None
    if version < MINIMUM_GMSH_VERSION:
        raise MissingOptionalDependency(
            "gmsh",
            GMSH_INSTALL_COMMAND,
            reason="API version {0}.{1} is installed and at least {2} is "
            "needed".format(version[0], version[1], needed),
        )
    # It is the API that matters, since the calls used here were changed in gmsh 4.7
    # and the option names again afterwards. An installation too old to report its API
    # version is refused rather than accepted, which is what checking __version__ did

    return gmsh


def has_gmsh():
    """
    Check whether a usable gmsh installation is available.

    Returns
    -------
    bool
        True when `require_gmsh` would succeed.
    """
    try:
        require_gmsh()
    except MissingOptionalDependency:
        return False

    return True
