"""
Solve linear elasticity with FEniCS on a microstructure GMMD has meshed.

GMMD writes a mesh FEniCS reads when it is asked for the xdmf format, from a deck with
``Formats [xdmf]`` or from the command line:

    geommicgen-mesh mic.yaml --to xdmf

The file holds the cells and the phase of each, which dolfinx reads as the mesh and as
a set of cell tags. This script gives every phase a Young modulus of its own and
stretches the cell along x: each face through the origin is held against moving through
itself, the face opposite the first is moved along x by the strain times the length of
the cell, and the other faces are free. It prints the cells it read, how many of them
each phase holds, and the average of the stress along x over the cell, as JSON.

    python fenics_elasticity.py mesh.xdmf --young 1=1e3 2=1e4 --poisson 0.3

It runs with dolfinx, installed from conda-forge (the package fenics-dolfinx) or from
the dolfinx container images; GMMD itself is not needed to run it.

A two dimensional mesh is analysed in plane strain. With one material the solution is
the homogeneous one, which linear elements represent exactly, so the average stress is
then known in closed form: E / (1 - nu^2) times the strain in plane strain, and E times
the strain in three dimensions.
"""

import argparse
import json

import numpy as np
import ufl
from dolfinx import fem, mesh as dmesh
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import XDMFFile
from mpi4py import MPI


def parse_arguments(argv=None):
    """Read the mesh file, the materials and the strain from the command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("mesh", help="xdmf file GMMD wrote")
    parser.add_argument(
        "--young",
        nargs="+",
        default=[],
        metavar="PHASE=E",
        help="Young modulus of a phase; a phase not named takes 1e3",
    )
    parser.add_argument("--poisson", type=float, default=0.3, help="Poisson ratio")
    parser.add_argument("--strain", type=float, default=0.1, help="stretch along x")
    return parser.parse_args(argv)


def main(argv=None):
    """Read the mesh and its phases, solve, and print what was found."""
    arguments = parse_arguments(argv)
    young = {int(i_pair.split("=")[0]): float(i_pair.split("=")[1])
             for i_pair in arguments.young}

    with XDMFFile(MPI.COMM_WORLD, arguments.mesh, "r") as xdmf:
        domain = xdmf.read_mesh(name="Grid")
        tdim = domain.topology.dim
        domain.topology.create_connectivity(tdim, 0)
        phases = xdmf.read_meshtags(domain, name="Grid")
    # GMMD writes one grid, named as meshio names it, holding the phase of every cell
    # beside the cells themselves

    materials = fem.functionspace(domain, ("DG", 0))
    modulus = fem.Function(materials)
    modulus.x.array[:] = 1.0e3
    for i_phase in np.unique(phases.values):
        cells = phases.find(i_phase)
        modulus.x.array[materials.dofmap.list[cells].ravel()] = young.get(
            int(i_phase), 1.0e3
        )
    nu = arguments.poisson
    lam = modulus * nu / ((1 + nu) * (1 - 2 * nu))
    mu = modulus / (2 * (1 + nu))
    # One value per cell, taken from the tag of its phase

    space = fem.functionspace(domain, ("Lagrange", 1, (domain.geometry.dim,)))
    u, v = ufl.TrialFunction(space), ufl.TestFunction(space)

    def stress(w):
        strain = ufl.sym(ufl.grad(w))
        return lam * ufl.tr(strain) * ufl.Identity(len(w)) + 2 * mu * strain

    a = ufl.inner(stress(u), ufl.sym(ufl.grad(v))) * ufl.dx
    zero = fem.Constant(domain, np.zeros(domain.geometry.dim))
    rhs = ufl.inner(zero, v) * ufl.dx

    coordinates = domain.geometry.x[:, : domain.geometry.dim]
    lower, upper = coordinates.min(axis=0), coordinates.max(axis=0)
    tol = 1e-8 * np.max(upper - lower)
    # A face is found within a length relative to the cell. The default of np.isclose
    # is a length of 1e-8, which in a micrometre cell took in nodes inside it, and in a
    # large one missed nodes of the face
    fdim = tdim - 1
    conditions = []
    for i_dir in range(domain.geometry.dim):
        facets = dmesh.locate_entities_boundary(
            domain,
            fdim,
            lambda x, d=i_dir: np.isclose(x[d], lower[d], rtol=0, atol=tol),
        )
        dofs = fem.locate_dofs_topological(space.sub(i_dir), fdim, facets)
        conditions.append(fem.dirichletbc(0.0, dofs, space.sub(i_dir)))
    facets = dmesh.locate_entities_boundary(
        domain, fdim, lambda x: np.isclose(x[0], upper[0], rtol=0, atol=tol)
    )
    dofs = fem.locate_dofs_topological(space.sub(0), fdim, facets)
    stretch = arguments.strain * (upper[0] - lower[0])
    conditions.append(fem.dirichletbc(stretch, dofs, space.sub(0)))
    # Rollers on the faces through the origin, and the opposite face along x moved by
    # the strain times the length of the cell

    problem = LinearProblem(
        a,
        rhs,
        bcs=conditions,
        petsc_options_prefix="gmmd_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    )
    solution = problem.solve()

    volume = fem.assemble_scalar(fem.form(1.0 * ufl.dx(domain=domain)))
    mean_stress = fem.assemble_scalar(fem.form(stress(solution)[0, 0] * ufl.dx)) / volume
    values, counts = np.unique(phases.values, return_counts=True)
    print(
        json.dumps(
            {
                "cells": int(domain.topology.index_map(tdim).size_local),
                "phases": {str(int(i)): int(n) for i, n in zip(values, counts)},
                "dim": int(domain.geometry.dim),
                "mean_stress_xx": float(mean_stress),
                "finite": bool(np.all(np.isfinite(solution.x.array))),
            }
        )
    )


if __name__ == "__main__":
    main()
