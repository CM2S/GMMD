# Changelog

The notable changes to GMMD, newest first. Versions follow [Semantic Versioning](https://semver.org). Each release is archived on Zenodo, under the DOI [10.5281/zenodo.24077934](https://doi.org/10.5281/zenodo.24077934).

## Unreleased

### Added

- The Zenodo DOI, in `CITATION.cff` and the readme ([#17](https://github.com/CM2S/GMMD/pull/17)).
- This changelog.

### Changed

- The documentation is titled GMMD. Its API pages no longer document the test modules, and document the classes that `particleclasses` and `translators` re-export only once ([#18](https://github.com/CM2S/GMMD/pull/18)).

### Fixed

- The documentation badge, which GitHub could not display ([#17](https://github.com/CM2S/GMMD/pull/17)).

### Removed

- `UML.io`, a diagram from 2021 that nothing referred to ([#17](https://github.com/CM2S/GMMD/pull/17)).

## [0.1.0](https://github.com/CM2S/GMMD/releases/tag/v0.1.0) - 2026-11-09

First tagged release.

- **Particles:** disks and ellipses in 2D; spheres, ellipsoids, cylinders and long cylindrical fibers in 3D.
- **Generation:** time-driven molecular dynamics with isokinetic thermostats. A fixed seed gives the same microstructure on every run.
- **Input:** a YAML input file, checked in full before anything is generated, with every problem reported at its place in the file.
- **Microstructure file:** a run first writes `mic.json`, a human-readable, tool-agnostic list of the particles, one per line, with the GMMD commit and library versions that made it. Meshing and analysis read it.
- **Meshing:** voxel grids, and Gmsh meshes conforming to the particles, translated for LINKS, CRATE, Abaqus and FEniCS.
- **Analysis:** 2-point correlation, Ripley's K, nearest neighbour distances and Voronoi metrics.

Coming from an earlier version: an old `.mdsim` input data file still runs, and `geommicgen-convert-mdsim` converts it into an input file.
