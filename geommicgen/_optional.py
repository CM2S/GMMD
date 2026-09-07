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
# Earliest release of gmsh whose API matches the one used here

GMSH_INSTALL_COMMAND = "pip install 'geommicgen[gmsh]'"


def _version_tuple(version):
    """
    Convert a version string into a tuple of integers.

    Parameters
    ----------
    version: str
        Version string, such as *"4.15.2"*.

    Returns
    -------
    tuple
        Tuple with the numeric components of the version.
    """
    components = []
    for i_component in str(version).split("."):
        digits = ""
        for i_char in i_component:
            if not i_char.isdigit():
                break
            digits += i_char
        if digits == "":
            break
        components.append(int(digits))
    # Trailing labels such as the ones in a development release are discarded

    return tuple(components)


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

    version = getattr(gmsh, "__version__", None)
    if version is not None and _version_tuple(version) < MINIMUM_GMSH_VERSION:
        raise MissingOptionalDependency(
            "gmsh",
            GMSH_INSTALL_COMMAND,
            reason="version {0} is installed and at least {1} is needed".format(
                version, ".".join(str(i_part) for i_part in MINIMUM_GMSH_VERSION)
            ),
        )
    # The API calls used here were changed in gmsh 4.7 and the option names again
    # afterwards, so an old installation fails in ways that are hard to read

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


def require_gmsh2links():
    """
    Get the mesh reader of gmsh2links, checking that the package is installed.

    Returns
    -------
    callable
        The *readMesh* function of gmsh2links.

    Raises
    ------
    MissingOptionalDependency:
        If gmsh2links is not installed.
    """
    try:
        from gmsh2links.main import readMesh
    except ImportError:
        raise MissingOptionalDependency(
            "gmsh2links",
            "pip install git+https://github.com/CM2S/Utilities.git#subdirectory=gmsh",
        ) from None
    # gmsh2links is not published on PyPI, so it cannot be declared as an extra
    # without making this package itself unpublishable

    return readMesh
