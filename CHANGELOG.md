# Changelog

The notable changes to GMMD, newest first. Versions follow [Semantic Versioning](https://semver.org). Each release is archived on Zenodo, under the DOI [10.5281/zenodo.24077934](https://doi.org/10.5281/zenodo.24077934).

## Unreleased

### Added

- The Zenodo DOI, in `CITATION.cff` and the readme ([#17](https://github.com/CM2S/GMMD/pull/17)).
- This changelog.
- Random sequential adsorption, a second method of generation, asked for with a `random_sequential_adsorption` block under `generation` ([#20](https://github.com/CM2S/GMMD/pull/20)).
- Squares, in two dimensions ([#20](https://github.com/CM2S/GMMD/pull/20)).
- `generation.inner_phase_max_step`, the positions tried for each particle of a phase placed inside another, which were 200 ([#20](https://github.com/CM2S/GMMD/pull/20)).
- Two analyses: `rsa_history`, the volume fraction a random sequential adsorption reached at every step, and `simulation_gif`, a GIF of a two-dimensional generation ([#20](https://github.com/CM2S/GMMD/pull/20)).
- A user guide in the documentation, with tutorials and the options of the input file, whose inputs are in `examples/` ([#21](https://github.com/CM2S/GMMD/pull/21)).
- A banner of generated microstructures in the readme, made by the inputs and scripts in `docs/images/src` ([#22](https://github.com/CM2S/GMMD/pull/22)).

### Changed

- The documentation is titled GMMD. Its API pages no longer document the test modules, and document the classes that `particleclasses` and `translators` re-export only once ([#18](https://github.com/CM2S/GMMD/pull/18)).
- The speed up schemes are a package, `geommicgen.micgenmethod.speed_up_schemes`. Those of the molecular dynamics simulation are imported from `geommicgen.micgenmethod.speed_up_schemes.MD_speed_up_schemes` ([#20](https://github.com/CM2S/GMMD/pull/20)).
- The two-dimensional plots of the particles, the final configuration and the frames of the GIF, no longer draw ticks and labels ([#22](https://github.com/CM2S/GMMD/pull/22)).
- The plots share one style, a serif face and the colours of Okabe and Ito. The histories of a molecular dynamics generation have axis labels, titles and legends, and the two-dimensional plots of the particles draw them opaque on a grey matrix ([#25](https://github.com/CM2S/GMMD/pull/25)).

### Fixed

- The documentation badge, which GitHub could not display ([#17](https://github.com/CM2S/GMMD/pull/17)).
- Ellipsoids kept `min_distance` apart could end up closer than that between their axes. They are now dilated by the amount that keeps the whole gap, as ellipses already were ([#20](https://github.com/CM2S/GMMD/pull/20)).

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
