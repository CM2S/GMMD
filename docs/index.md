% The sidebar shows the logo, so the page starts after the readme's logo
```{include} ../readme.md
:relative-images:
:start-after: docs/images/logo/mark.svg)
```

```{toctree}
:maxdepth: 1
:caption: Tutorials
:hidden:

tutorials/MD_Ellipses/MD_Ellipses
tutorials/MD_Disks/MD_Disks
tutorials/RSA_Fibers/RSA_Fibers
tutorials/MD_Inner_Outer_Disks/MD_Inner_Outer_Disks
tutorials/MD_Ellipsoids/MD_Ellipsoids
tutorials/MD_Spheres_Cylinders/MD_Spheres_Cylinders
tutorials/RSA_Squares/RSA_Squares
tutorials/Voronoi/Voronoi
```

```{toctree}
:maxdepth: 2
:caption: Input file
:hidden:

input_file/index
```

```{toctree}
:maxdepth: 1
:caption: Reference
:hidden:

API <modules>
Papers <papers>
```
