---
title: 'GMMD: A Python package for geometric microstructure generation'
tags:
  - Python

authors:
  - name: Some author
    orcid: 0000-0000-0000-0000
    equal-contrib: true
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
  - name: Author Without ORCID
    equal-contrib: true # (This is how you can denote equal contributions between multiple authors)
    affiliation: 2
  - name: Author with no affiliation
    corresponding: true # (This is how to denote the corresponding author)
    affiliation: 3
  - given-names: Ludwig
    dropping-particle: van
    surname: Beethoven
    affiliation: 3
affiliations:
 - name: Faculty of Engineering, University of Porto, Porto, Portugal
   index: 1
   ror: 00hx57361
 - name: Institute of Science and Innovation in Mechanical and Industrial Engineering, Porto, Portugal 
   index: 2
 - name: Independent Researcher, Country
   index: 3
date: 13 August 2017
bibliography: paper.bib

# Optional fields for papers that are part of a joint submission.
# For example, submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
#
# If you are not making a joint submission you should remove these lines.
#
#aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
#aas-journal: Astrophysical Journal <- The name of the AAS journal.


# Note: the outline of this paper was copied from the JOSS example paper. To be changed later.
# Work log:
#  - First draft of the summary and statement of need

---


# Summary
GMMD is an open-source Python tool to generate microstructures. It can generate single or multiphase 2D or 3D microstructures with varying RVE (representative volume element) dimensions. At the moment, the particle (or void) shapes available are disks, ellipses, squares, spheres, ellipsoids, fibers and cylinders. Moreover, the code was designed so that further shapes can be easily added. GMMD can, for now, use one of two methods for generating the RVE: molecular dynamics and random sequential addition.
GMMD can export the final microstructure configuration (PDF for 2D, VTK for 2D and 3D) and can generate 2D simulation GIFs. It also includes built-in tools for performing statistical analyses on the microstructure.
After the generation procedure, the RVE can be discretized in a suitable finite element mesh in order to perform microscale analyses through computational homogenization.



# Statement of need

The undestanding of the behaviour of complex materials is fundamental to use them in the limits of its capabilities, whether the material is natural and traditional, such as wood and soils, or modern synthetics, as is a fiber reinforced polymer, for example.

These complex materials often are heterogenous, as is nearly every material at a small enough schale. Simulating an entire structural part considering all its heterogeneities, often very small comparative to the part size, can be very time consuming and extremely impractical.

Thus, a multiscale approach emerged, where if one knows the microstructural features of the material:

1. the properties of the constituents,
2. the properties of the interface,
3. and the geometry of each phase,

one can obtain its macroscale properties using a process known as computational homogenization.
Thus, this requires the generation of a representative volume element (RVE), that is, in broad terms, a small volume element representative of the entire microstructure in an average sense [@HILL1963357].

GMMD enters as a solution for generating RVEs for particle reinforced materials, saving researchers and designers much time, especially if a large number of samples is required.
It also facilitates AI material design, since it can generate the datasets to train machine learning models. This can replace slow, iterative design cycles with a much faster, automated process.

GMMD is, thus, a python package that generates microstrucutre geometry depending on the user input, such as particle (or void) shape and a broad range of microstructure descriptors following different statistical distributions.

The generation of the microstructure is not based on the physical process of which it arised, it is purely geometric. GMMD can, for now, use one of two methods for generating the RVE: molecular dynamics and random sequential addition.

@VILACHA2021104069 presents the theory behind the molecular dynamics simulation, while the Numerical assessment and statistical analysis of the microstructures obtained via molecular dynamics is provided in the article @FERREIRA2022104068.

!!Talk about other softwares and how this software brings something new and useful.
(Softwares que o gemini indicou, por ordem de prioridade de pesquisa.)
- Neper: polycristal (verificar de o GMMD cria microestruturas policrsitalinas)
- PoreSpy: Porous / Granular Media
- Kanapy: Polycrystals / Granular (está no joss)
- DREAM.3D
- TexGen: Woven / Braided Composites
- MicroStructPy: Particulate / Inclusions / Foams



# Mathematics

Single dollars ($) are required for inline mathematics e.g. $f(x) = e^{\pi/x}$

Double dollars make self-standing equations:

$$\Theta(x) = \left\{\begin{array}{l}
0\textrm{ if } x < 0\cr
1\textrm{ else}
\end{array}\right.$$

You can also use plain \LaTeX for equations
\begin{equation}\label{eq:fourier}
\hat f(\omega) = \int_{-\infty}^{\infty} f(x) e^{i\omega x} dx
\end{equation}
and refer to \autoref{eq:fourier} from text.


# AI usage disclosure

No generative AI tools were used in the development of this software, the writing
of this manuscript, or the preparation of supporting materials.

# Acknowledgements

We acknowledge contributions from Brigitta Sipocz, Syrtis Major, and Semyeong
Oh, and support from Kathryn Johnston during the genesis of this project.

# References