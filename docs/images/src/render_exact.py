"""Render the exact shapes of the particles of a GMMD microstructure file, with no mesh.

    pvbatch render_exact.py mic.json OUT.png [--size 3200] [--line-width 2]
        [--colors 1=#0072B2,2=#D55E00] [--matrix-opacity 0.18] [--res 160]

Each particle and each of its periodic images is drawn from its parameters in mic.json
and clipped to the RVE, in 3D with a flat cap where it is cut. The matrix is a frosted
box in 3D and a grey rectangle in 2D. The background is transparent. The image is
square in 3D and has the proportions of the RVE in 2D; compose.py crops and scales it.
"""
import argparse
import itertools
import json
import math

from paraview.simple import (
    PVTrivialProducer, Show, GetActiveViewOrCreate, SaveScreenshot, ResetCamera, Render,
)
from vtkmodules.vtkCommonDataModel import vtkPlane, vtkPlaneCollection, vtkBox
from vtkmodules.vtkCommonTransforms import vtkTransform
from vtkmodules.vtkFiltersCore import (
    vtkAppendPolyData, vtkCleanPolyData, vtkPolyDataNormals, vtkTriangleFilter,
    vtkClipPolyData,
)
from vtkmodules.vtkFiltersGeneral import vtkClipClosedSurface, vtkTransformPolyDataFilter
from vtkmodules.vtkFiltersSources import (
    vtkSphereSource, vtkCylinderSource, vtkCubeSource, vtkRegularPolygonSource,
    vtkOutlineSource, vtkPlaneSource,
)

OKABE_ITO = ["#0072B2", "#D55E00", "#009E73", "#E69F00", "#56B4E9", "#CC79A7"]

p = argparse.ArgumentParser()
p.add_argument("mic")
p.add_argument("out")
p.add_argument("--size", type=int, default=3200)
p.add_argument("--line-width", type=float, default=2)
p.add_argument("--colors", default="")
p.add_argument("--azimuth", type=float, default=40)
p.add_argument("--elevation", type=float, default=28)
p.add_argument("--matrix-opacity", type=float, default=0.18)
p.add_argument("--res", type=int, default=160)
a = p.parse_args()


def rgb(h):
    h = h.lstrip("#")
    return [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]


mic = json.load(open(a.mic))
L = mic["rve_dims"]
dim = len(L)
phases = sorted({int(q["phase"]) for q in mic["particles"]})
colors = {k: rgb(OKABE_ITO[(k - 1) % len(OKABE_ITO)]) for k in phases}
for item in filter(None, a.colors.split(",")):
    k, h = item.split("=")
    colors[int(k)] = rgb(h)


def transformed(src, t):
    f = vtkTransformPolyDataFilter()
    f.SetInputConnection(src.GetOutputPort())
    f.SetTransform(t)
    f.Update()
    return f.GetOutput()


def shape(q, shift):
    """Closed surface (3D) or polygon (2D) of particle q moved by shift."""
    c = [q["center"][i] + shift[i] for i in range(len(shift))]
    s = q["shape"]
    t = vtkTransform()
    if s in ("Sphere", "Ellipsoid"):
        src = vtkSphereSource()
        src.SetThetaResolution(a.res)
        src.SetPhiResolution(a.res // 2)
        src.SetRadius(1)
        t.Translate(*c)
        if s == "Sphere":
            t.Scale(q["r"], q["r"], q["r"])
        else:
            t.RotateWXYZ(math.degrees(q["angle"]), *q["rotation_axis"])
            t.Scale(q["axis_1"] / 2, q["axis_2"] / 2, q["axis_3"] / 2)
        return transformed(src, t)
    if s == "Cylinder":
        src = vtkCylinderSource()
        src.SetResolution(2 * a.res)
        src.SetRadius(q["r_cyl"])
        src.SetHeight(q["length"])
        src.CappingOn()
        th, ph = q["polar_angle"], q["azimuth_angle"]
        # The source runs along y; turned onto z, then by the polar angle about the
        # direction in which the azimuth grows, as the mesher turns it
        t.Translate(*c)
        t.RotateWXYZ(math.degrees(th), -math.sin(ph), math.cos(ph), 0)
        t.RotateX(90)
        return transformed(src, t)
    if s == "CylindricalFiber":
        src = vtkCylinderSource()
        src.SetResolution(2 * a.res)
        src.SetRadius(q["r"])
        d = q["direction"]
        src.SetHeight(L[d] * 1.2)
        src.CappingOn()
        pos = list(c)
        pos.insert(d, L[d] / 2)
        t.Translate(*pos)
        # the source runs along y
        if d == 0:
            t.RotateZ(90)
        elif d == 2:
            t.RotateX(90)
        return transformed(src, t)
    src = vtkRegularPolygonSource()
    src.SetGeneratePolygon(1)
    src.SetGeneratePolyline(0)
    src.SetRadius(1)
    t.Translate(c[0], c[1], 0)
    if s == "Square":
        src.SetNumberOfSides(4)
        t.RotateZ(45)
        t.Scale(q["side"] / math.sqrt(2), q["side"] / math.sqrt(2), 1)
    else:
        src.SetNumberOfSides(4 * a.res)
        if s == "Disk":
            t.Scale(q["r"], q["r"], 1)
        else:
            t.RotateZ(math.degrees(q["angle"]))
            t.Scale(q["major_axis"] / 2, q["minor_axis"] / 2, 1)
    return transformed(src, t)


def reach(q):
    if "r_cyl" in q:
        return math.hypot(q["r_cyl"], q["length"] / 2)
    for k in ("r", "axis_1", "major_axis"):
        if k in q:
            return q[k] if k == "r" else q[k] / 2
    return q["side"] / math.sqrt(2)


def images(q):
    """Shifts by one RVE length that bring the particle's bounding sphere into the RVE."""
    n = len(q["center"])
    dims = [L[i] for i in range(dim)]
    if q["shape"] == "CylindricalFiber":
        dims = [L[i] for i in range(dim) if i != q["direction"]]
    r = reach(q)
    for k in itertools.product((-1, 0, 1), repeat=n):
        c = [q["center"][i] + k[i] * dims[i] for i in range(n)]
        if all(-r < c[i] < dims[i] + r for i in range(n)):
            yield [k[i] * dims[i] for i in range(n)]


def box_planes():
    pc = vtkPlaneCollection()
    for i in range(3):
        for side, normal in ((0, -1), (L[i], 1)):
            pl = vtkPlane()
            o = [0, 0, 0]
            o[i] = side
            nv = [0, 0, 0]
            nv[i] = -normal
            pl.SetOrigin(*o)
            pl.SetNormal(*nv)
            pc.AddItem(pl)
    return pc


view = GetActiveViewOrCreate("RenderView")
shape_px = [a.size, a.size] if dim == 3 else [a.size, round(a.size * L[1] / L[0])]
view.ViewSize = shape_px
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


def show_poly(poly, color, opacity=1.0, flat=False, line=None):
    tp = PVTrivialProducer()
    tp.GetClientSideObject().SetOutput(poly)
    d = Show(tp, view)
    d.SetRepresentationType("Surface")
    d.ColorArrayName = [None, ""]
    d.AmbientColor = d.DiffuseColor = color
    d.Opacity = opacity
    if flat:
        d.Ambient, d.Diffuse, d.Specular = 1.0, 0.0, 0.0
    else:
        d.Ambient, d.Diffuse, d.Specular, d.SpecularPower = 0.15, 0.85, 0.35, 50
    if line:
        d.LineWidth = line
    return d


for k in phases:
    app = vtkAppendPolyData()
    for q in mic["particles"]:
        if int(q["phase"]) != k:
            continue
        for sh in images(q):
            poly = shape(q, sh)
            if dim == 3:
                tri = vtkTriangleFilter()
                tri.SetInputData(poly)
                clean = vtkCleanPolyData()
                clean.SetInputConnection(tri.GetOutputPort())
                clip = vtkClipClosedSurface()
                clip.SetInputConnection(clean.GetOutputPort())
                clip.SetClippingPlanes(box_planes())
                clip.SetGenerateFaces(1)
                clip.Update()
                poly = clip.GetOutput()
            else:
                b = vtkBox()
                b.SetBounds(0, L[0], 0, L[1], -1, 1)
                clip = vtkClipPolyData()
                clip.SetInputData(poly)
                clip.SetClipFunction(b)
                clip.InsideOutOn()
                clip.Update()
                poly = clip.GetOutput()
            if poly.GetNumberOfPoints():
                app.AddInputData(poly)
    app.Update()
    if dim == 3:
        nrm = vtkPolyDataNormals()
        nrm.SetInputConnection(app.GetOutputPort())
        nrm.SetFeatureAngle(40)
        nrm.SplittingOn()
        nrm.Update()
        show_poly(nrm.GetOutput(), colors[k])
    else:
        show_poly(app.GetOutput(), colors[k], flat=True)

MATRIX = rgb("#E4E4E4")
if dim == 3:
    # The faces stand a little outside the cut faces of the particles, which would
    # otherwise lie in their planes and flicker through them
    e = 0.003 * max(L)
    cube = vtkCubeSource()
    cube.SetBounds(-e, L[0] + e, -e, L[1] + e, -e, L[2] + e)
    cube.Update()
    show_poly(cube.GetOutput(), MATRIX, a.matrix_opacity)
    out = vtkOutlineSource()
    out.SetBounds(-e, L[0] + e, -e, L[1] + e, -e, L[2] + e)
    out.Update()
    show_poly(out.GetOutput(), rgb("#7A7A7A"), 1.0, flat=True, line=a.line_width)
else:
    pl = vtkPlaneSource()
    pl.SetOrigin(0, 0, -0.01)
    pl.SetPoint1(L[0], 0, -0.01)
    pl.SetPoint2(0, L[1], -0.01)
    pl.Update()
    show_poly(pl.GetOutput(), MATRIX, flat=True)

ResetCamera(view)
if dim == 3:
    c = [x / 2 for x in L]
    az, el = math.radians(a.azimuth), math.radians(a.elevation)
    d = [math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)]
    view.CameraFocalPoint = c
    view.CameraPosition = [c[i] + 5 * d[i] for i in range(3)]
    view.CameraViewUp = [0, 0, 1]
    view.CameraParallelProjection = 0
    ResetCamera(view)
else:
    view.CameraParallelProjection = 1
    view.CameraFocalPoint = [L[0] / 2, L[1] / 2, 0]
    view.CameraPosition = [L[0] / 2, L[1] / 2, 5]
    view.CameraViewUp = [0, 1, 0]
    view.CameraParallelScale = L[1] / 2
Render(view)
SaveScreenshot(a.out, view, ImageResolution=shape_px, TransparentBackground=1)
print("wrote", a.out)
