# Changelog

The notable changes to GMMD, newest first. Versions follow [Semantic Versioning](https://semver.org). Each release is archived on Zenodo, under the DOI [10.5281/zenodo.24077934](https://doi.org/10.5281/zenodo.24077934).

## Unreleased

### Changed

- The readme lists the five authors of GMMD under "Authors", where it named four of them as contributors.

## [0.2.0](https://github.com/CM2S/GMMD/releases/tag/v0.2.0) - 2026-11-12

Random sequential adsorption and squares, a user guide, and the first release on PyPI.

### Added

- The Zenodo DOI, in `CITATION.cff` and the readme ([#17](https://github.com/CM2S/GMMD/pull/17)).
- This changelog.
- Random sequential adsorption, a second method of generation, asked for with a `random_sequential_adsorption` block under `generation` ([#20](https://github.com/CM2S/GMMD/pull/20)).
- Squares, in two dimensions ([#20](https://github.com/CM2S/GMMD/pull/20)).
- `generation.inner_phase_max_step`, the positions tried for each particle of a phase placed inside another, which were 200 ([#20](https://github.com/CM2S/GMMD/pull/20)).
- Two analyses: `rsa_history`, the volume fraction a random sequential adsorption reached at every step, and `simulation_gif`, a GIF of a two-dimensional generation ([#20](https://github.com/CM2S/GMMD/pull/20)).
- A user guide in the documentation, with tutorials and the options of the input file, whose inputs are in `examples/` ([#21](https://github.com/CM2S/GMMD/pull/21)).
- A banner of generated microstructures in the readme, made by the inputs and scripts in `docs/images/src` ([#22](https://github.com/CM2S/GMMD/pull/22)).
- A logo, in the readme and the documentation, with its files in `docs/images/logo`, in colour and in black and white ([#28](https://github.com/CM2S/GMMD/pull/28)).
- GMMD on PyPI, as `geommicgen`, installed with `pip install geommicgen` or `pipx install geommicgen`. Publishing a release on GitHub uploads it to PyPI, by `.github/workflows/release.yaml` ([#30](https://github.com/CM2S/GMMD/pull/30)).

### Changed

- The documentation is titled GMMD. Its API pages no longer document the test modules, and document the classes that `particleclasses` and `translators` re-export only once ([#18](https://github.com/CM2S/GMMD/pull/18)).
- The speed up schemes are a package, `geommicgen.micgenmethod.speed_up_schemes`. Those of the molecular dynamics simulation are imported from `geommicgen.micgenmethod.speed_up_schemes.MD_speed_up_schemes` ([#20](https://github.com/CM2S/GMMD/pull/20)).
- The two-dimensional plots of the particles, the final configuration and the frames of the GIF, no longer draw ticks and labels ([#22](https://github.com/CM2S/GMMD/pull/22)).
- The plots share one style, a serif face and the colours of Okabe and Ito. The histories of a molecular dynamics generation have axis labels, titles and legends, and the two-dimensional plots of the particles draw them opaque on a grey matrix ([#25](https://github.com/CM2S/GMMD/pull/25)).
- The plots of the nearest neighbour distances and of the two point correlation say that the distances are over the mean radius, and the frames of the GIF of a generation are drawn at 150 dots per inch, which brings them back to their size before #25 ([#26](https://github.com/CM2S/GMMD/pull/26)).
- The inputs of the tutorials are seeded, so they give the microstructures of the figures. `MD_Ellipsoids` fills 30 % with larger ellipsoids, so that its plots show the thermostat and the adaptive time step at work. The figures of the tutorials are made again with the current plots, and the three-dimensional ones are drawn as in the banner ([#26](https://github.com/CM2S/GMMD/pull/26)).
- The examples `2D_example_ellipses_vf_50_n_100`, `3D_example_ellipsoids_vf_10_r_1_7` and `RSA_disk` are seeded too, so the tests that run them, and the coverage they measure, give the same result on every run ([#29](https://github.com/CM2S/GMMD/pull/29)).
- The readme's images are full URLs, so that PyPI shows them, and its installation starts from PyPI. The package gives its license as the SPDX expression `BSD-3-Clause`, and classifiers for the Python versions and systems it is tested on ([#30](https://github.com/CM2S/GMMD/pull/30)).

### Fixed

- The documentation badge, which GitHub could not display ([#17](https://github.com/CM2S/GMMD/pull/17)).
- Ellipsoids kept `min_distance` apart could end up closer than that between their axes. They are now dilated by the amount that keeps the whole gap, as ellipses already were ([#20](https://github.com/CM2S/GMMD/pull/20)).
- The plot of the kinetic energy drew the kinetic energy at the reference temperature one step early ([#25](https://github.com/CM2S/GMMD/pull/25)).
- The gmsh mesher refused the mesh of 194 ellipsoids that random sequential adsorption placed, because the cells covered only part of the RVE. Gmsh had left the surface of one ellipsoid without elements, and only warned. The mesher now meshes the model again, refined around such a surface, as it already did for a surface gmsh raised an error over ([#32](https://github.com/CM2S/GMMD/pull/32)).

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
