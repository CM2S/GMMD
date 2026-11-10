"""Render a GMMD mesh (the .vtu file a run writes beside the mesh) with its elements.

    pvbatch render_mesh.py MESH.vtu OUT.png [--dim 3] [--size 3200] [--line-width 2]
        [--colors 1=#0072B2,2=#D55E00] [--matrix-opacity 0.25] [--edge-shade 0.4]

In 3D the matrix is drawn as the faces of the RVE, see-through, and each phase as the
surface of its elements. In 2D every element is drawn, the matrix in grey. The
background is transparent. The image is square in 3D and has the proportions of the RVE
in 2D; compose.py crops and scales it.
"""
import argparse
import math

from paraview.simple import (
    XMLUnstructuredGridReader, Threshold, ExtractSurface, Show, GetActiveViewOrCreate,
    SaveScreenshot, ResetCamera, Render,
)

OKABE_ITO = ["#0072B2", "#D55E00", "#009E73", "#E69F00", "#56B4E9", "#CC79A7"]
MATRIX, MATRIX_EDGES = "#E4E4E4", "#6A6A6A"

p = argparse.ArgumentParser()
p.add_argument("mesh")
p.add_argument("out")
p.add_argument("--dim", type=int, default=3)
p.add_argument("--size", type=int, default=3200)
p.add_argument("--line-width", type=float, default=2)
p.add_argument("--colors", default="")
p.add_argument("--matrix-opacity", type=float, default=0.25)
p.add_argument("--edge-shade", type=float, default=0.4)
p.add_argument("--azimuth", type=float, default=40)
p.add_argument("--elevation", type=float, default=28)
a = p.parse_args()


def rgb(h):
    h = h.lstrip("#")
    return [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]


def only(src, k):
    t = Threshold(Input=src, Scalars=["CELLS", "phase"])
    t.LowerThreshold, t.UpperThreshold = k - 0.5, k + 0.5
    return t


reader = XMLUnstructuredGridReader(FileName=[a.mesh])
reader.UpdatePipeline()
b = reader.GetDataInformation().GetBounds()
lo, hi = reader.CellData["phase"].GetRange()
phases = [k for k in range(int(lo), int(hi) + 1) if k != 0]
colors = {k: rgb(OKABE_ITO[(k - 1) % len(OKABE_ITO)]) for k in phases}
for item in filter(None, a.colors.split(",")):
    k, h = item.split("=")
    colors[int(k)] = rgb(h)

px = [a.size, a.size] if a.dim == 3 else [a.size, round(a.size * (b[3] - b[2]) / (b[1] - b[0]))]
view = GetActiveViewOrCreate("RenderView")
view.ViewSize = px
view.Background = [1, 1, 1]
try:
    view.UseColorPaletteForBackground = 0
except AttributeError:
    pass
view.OrientationAxesVisibility = 0
# No FXAA: on a transparent background it would blend the edges with white. The
# image is rendered large and scaled down by compose.py instead
view.UseFXAA = 0
view.DepthPeeling = 1
view.MaximumNumberOfPeels = 8


def show(src, color, edge, opacity=1.0):
    d = Show(src, view)
    d.SetRepresentationType("Surface With Edges")
    d.ColorArrayName = [None, ""]
    d.AmbientColor = d.DiffuseColor = color
    d.EdgeColor = edge
    d.LineWidth = a.line_width
    d.Opacity = opacity
    d.Ambient, d.Diffuse, d.Specular, d.SpecularPower = 0.15, 0.85, 0.25, 40


for k in phases:
    surface = ExtractSurface(Input=only(reader, k)) if a.dim == 3 else only(reader, k)
    show(surface, colors[k], [x * a.edge_shade for x in colors[k]])
if a.dim == 3:
    # The skin of the whole mesh is the faces of the RVE, where the matrix shows
    show(only(ExtractSurface(Input=reader), 0), rgb(MATRIX), rgb(MATRIX_EDGES),
         a.matrix_opacity)
else:
    show(only(reader, 0), rgb(MATRIX), rgb(MATRIX_EDGES))

c = [(b[0] + b[1]) / 2, (b[2] + b[3]) / 2, (b[4] + b[5]) / 2]
if a.dim == 3:
    # z up, seen from the (+x, +y, +z) side
    az, el = math.radians(a.azimuth), math.radians(a.elevation)
    d = [math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)]
    view.CameraFocalPoint = c
    view.CameraPosition = [c[i] + 5 * d[i] for i in range(3)]
    view.CameraViewUp = [0, 0, 1]
    ResetCamera(view)
else:
    view.CameraParallelProjection = 1
    view.CameraFocalPoint = c
    view.CameraPosition = [c[0], c[1], 5]
    view.CameraViewUp = [0, 1, 0]
    view.CameraParallelScale = (b[3] - b[2]) / 2
Render(view)
SaveScreenshot(a.out, view, ImageResolution=px, TransparentBackground=1)
print("wrote", a.out)
