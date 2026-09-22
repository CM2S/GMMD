__title__ = "geommicgen"
__author__ = "CM2S"
__copyright__ = "2026, CM2S"
__license__ = "BSD-3-Clause"
__version__ = "0.1.0"

# TODO: type annotations. The package has none, so a type checker infers every type
# and reports where an inferred list or array meets a library that annotates
# precisely (meshio, numpy), which is noise rather than a finding. Every function
# carries a numpydoc docstring that states its types informally, so annotating is a
# transcription. Start with the contracts the rest is written against -- Mesh,
# Mesher, SolverWriter, MeshJob -- and leave the simulation internals for last.
# After the JOSS submission: it touches every file and buys a reviewer nothing.
