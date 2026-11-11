# Sources of the figures

The banner of the readme (`docs/images/banner.png`) and the figure of the JOSS paper are made from the inputs in this folder.
`make_figures.sh` runs each input, renders each panel with ParaView, and lays the panels out.
The inputs are seeded, so every run gives the same microstructures.

The logo (`docs/images/logo`) is made from `logo.yaml` by `make_logo.sh`, which runs GMMD and then `make_logo.py`.
The mark is a periodic cell of the particles of `logo.yaml` on a rounded tile, and the logo is the mark with the name GMMD at its right.
Each is in colour for light grounds (`mark.svg`, `logo.svg`) and for dark grounds (`-on-dark`), and in one colour, black (`-black`) and white (`-white`).
