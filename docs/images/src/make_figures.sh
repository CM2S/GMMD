#!/usr/bin/env bash
# Remake the banner of the readme and the figure of the JOSS paper from the inputs here.
# Needs GMMD with Gmsh, ParaView's pvbatch, and Python with Pillow and NumPy; pngquant
# makes the files smaller. Run it from this folder; it starts ./build afresh and works
# there, where each input is copied, since GMMD writes the results beside the input.
set -euo pipefail
PVBATCH=${PVBATCH:-pvbatch}
PYTHON=${PYTHON:-python}
rm -rf build
mkdir build
cd build
for n in disk_ell squares sph_ell cyl_sph poly_sph aligned_ell fibres rsa_sph rect2d; do
    cp ../$n.yaml .
    geommicgen $n.yaml
done

exact() { "$PVBATCH" ../render_exact.py "$1/mic_0/mic.json" "$1.png" "${@:2}"; }
exact disk_ell --size 2800
exact squares --size 2800 --colors 1=#009E73
exact sph_ell
exact cyl_sph --colors 1=#0072B2,2=#E69F00
exact poly_sph
exact aligned_ell --colors 1=#CC79A7
exact fibres --colors 1=#56B4E9
exact rect2d --size 4000
"$PVBATCH" ../render_mesh.py rsa_sph/mic_0/meshes/rsa_sph_tetra4_epp6.vtu rsa_sph_mesh.png \
    --colors 1=#0072B2,2=#F0E442
"$PVBATCH" ../render_mesh.py rect2d/mic_0/meshes/rect2d_tri3_epp4.vtu rect2d_mesh.png \
    --dim 2 --size 4000 --edge-shade 0.55

"$PYTHON" ../compose.py banner.png --height 420 --gap 24 \
    disk_ell.png sph_ell.png cyl_sph.png fibres.png / \
    squares.png poly_sph.png aligned_ell.png rsa_sph_mesh.png
"$PYTHON" ../compose.py paper.png --height 900 --gap 48 rect2d_mesh.png sph_ell.png
# Without dithering, which would speckle the flat faces
pngquant --quality 85-100 --speed 1 --nofs --force --ext .png banner.png paper.png
