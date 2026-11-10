#!/usr/bin/env bash
# Remake the files of the logo in ../logo from logo.yaml.
# Needs GMMD; make_logo.py needs only the Python standard library. Run it from this
# folder; it starts ./build afresh and works there, since GMMD writes the results beside
# the input.
set -euo pipefail
PYTHON=${PYTHON:-python}
rm -rf build
mkdir build
cd build
cp ../logo.yaml .
geommicgen logo.yaml
"$PYTHON" ../make_logo.py logo/mic_0/mic.json ../../logo
