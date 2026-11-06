# Contributing to GMMD

## Reporting a problem

Open an issue at https://github.com/CM2S/GMMD/issues. For a bug, give:

- what you ran, and what you expected to happen;
- the full error message;
- the input file and, if the run wrote one, the microstructure file (`mic.json`), which records the GMMD commit and the versions of numpy, scipy and gmsh;
- your operating system and Python version.

## Getting help

Ask a question about using GMMD by opening an issue. A question the documentation does not answer is a gap in the documentation, and the issue helps us close it. You can also write to José Luís P. Vila-Chã (jvc@fe.up.pt).

## Contributing a change

1. Fork the repository and create a branch from `master`.
2. Install GMMD with the test extra: `pip install -e '.[test]'`.
3. Make the change, with a test that fails without it.
4. Run the test suite: `pytest geommicgen/tests`.
5. Update the documentation the change affects: the readme, `docs/`, and `examples/reference.yaml` for a new key of the input file.
6. Open a pull request against `master`. A maintainer reviews it and may ask for changes.

The rules in the readme's "Use of AI" section apply to every contribution.
