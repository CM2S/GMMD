---
title: 'GMG: A Python package for geometric microstructure generation'
tags:
  - Python

authors:
  - name: Adrian M. Price-Whelan
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
 - name: Lyman Spitzer, Jr. Fellow, Princeton University, United States
   index: 1
   ror: 00hx57361
 - name: Institution Name, Country
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
GMMD is an open-source Python tool to generate microstructures. It can generate single or multiphase 2D or 3D microstructures with varying RVE (representative volume element) dimensions. At the moment, the particle (or void) shapes available are disks, ellipses, squares, spheres, ellipsoids, fibers and cylinders. Moreover, the code was designed so that further shapes can be easily added. GMG can, for now, use one of two methods for generating the RVE: 1) molecular dynamics and 2) random sequential addition.
GMG can export the final microstructure configuration (PDF for 2D, VTK for 2D and 3D) and can generate 2D simulation GIFs. It also includes built-in tools for performing statistical analyses on the microstructure.
After the generation procedure, the RVE can be discretized in a suitable finite element mesh in order to perform microscale analyses through computational homogenization.





# Statement of need

Modeling the behaviour of materials and finding their mechanical properties is paramount for structural design. Some materials are easy to characterize (due to the past work of great minds), but the more complex the material, the harder is its characterization.

!!Examples of complex materials and their applications.

Every material is heteregenous at a small enough scale. Thus, in order to obtain the properties of certain materials, a multiscale aproach can be followed.
Lets take the example of a single ply of a composite material. At the microscale, one can distinguish matrix from fiber and even interface.
If we know:
    1. the material model and its properties for both matrix and reinforcement
    2. the interface model and properties
    3. the geometry of the fibers and the matrix
we can obtain the material properties at the ply scale (using a process known as computational homegenization).

On the other hand, if the material properties, the fiber and matrix geometry and the behaviour of the lamina is known (through physical testing), one can, perhaps, find a possible interface model.

GMG tackles the third point. It is a python package that generates microstrucutre geometry depending on the user input, such as particle (or void) shape, volume fraction and more descriptors. 

The generation of the microstructure is not based on the physical process of which it arised, it is purely geometric. GMG can, for now, use one of two methods for generating the RVE: 1) molecular dynamics and 2) random sequential addition.

It falls under the user to ensure that the virtual geometry, also known as RVE (representative volume element), is representative of the real geometry whose properties are to be determined.



!!Falta adicionar os artigos em que o software já foi utilizado.
!! Vale a pena fazer referência à tese do Zé?

!!Talk about other softwares and how this software brings something new and useful.
(Softwares que o gemini indicou, por ordem de prioridade de pesquisa.)
- Neper: polycristal (verificar de o GMMD cria microestruturas policrsitalinas)
- PoreSpy: Porous / Granular Media
- Kanapy: Polycrystals / Granular (está no joss)
- DREAM.3D
- TexGen: Woven / Braided Composites
- MicroStructPy: Particulate / Inclusions / Foams







# State of the field                                                                                                                  



# Software design



# Research impact statement


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