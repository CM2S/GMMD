"""Draw the logo of GMMD from a microstructure of disks and ellipses, as SVG files.

    python make_logo.py MIC_JSON OUT_DIR

The mark is a periodic cell of the particles on a rounded tile: a particle the cell cuts
comes back in at the opposite side. The logo is the mark with the name GMMD at its right.
Each is written in colour for light and for dark grounds, and in black and in white.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, fields, replace
from pathlib import Path

SIDE = 120  # side of the tile
CORNER = 26  # corner radius of the tile
CAP = 100  # cap height of the letters, in their own units
BLACK, WHITE = "#000000", "#FFFFFF"


@dataclass(frozen=True)
class Theme:
    """Colours of one version of the logo."""

    ink: str  # letters
    tile: str
    paper: str  # large ellipses
    ember: str  # large disk, and the dot of the G
    teal: str  # small disks and small ellipses


ON_LIGHT = Theme(ink="#17233B", tile="#17233B", paper="#F5EEE3", ember="#F0633A", teal="#2AA198")
ON_DARK = replace(ON_LIGHT, ink="#F5EEE3", tile="#1E2D4C")


def one_colour(colour: str) -> Theme:
    return Theme(**{field.name: colour for field in fields(Theme)})


@dataclass(frozen=True)
class Disk:
    """A disk in a periodic cell of side 1."""

    x: float
    y: float
    r: float
    phase: str

    @property
    def half_width(self) -> float:
        return self.r

    @property
    def half_height(self) -> float:
        return self.r

    @property
    def area(self) -> float:
        return math.pi * self.r ** 2


@dataclass(frozen=True)
class Ellipse:
    """An ellipse in a periodic cell of side 1."""

    x: float
    y: float
    a: float  # half of the major axis
    b: float  # half of the minor axis
    angle: float  # from the x axis to the major axis, in radians
    phase: str

    @property
    def half_width(self) -> float:
        return math.hypot(self.a * math.cos(self.angle), self.b * math.sin(self.angle))

    @property
    def half_height(self) -> float:
        return math.hypot(self.a * math.sin(self.angle), self.b * math.cos(self.angle))

    @property
    def area(self) -> float:
        return math.pi * self.a * self.b


Particle = Disk | Ellipse


@dataclass(frozen=True)
class Drawing:
    """SVG elements and the size of the box they are drawn in."""

    body: str
    width: float
    height: float


# SVG text
def n(value: float) -> str:
    """A coordinate as short text."""
    return f"{value:.2f}".rstrip("0").rstrip(".")


def circle(x: float, y: float, r: float, fill: str) -> str:
    return f'<circle cx="{n(x)}" cy="{n(y)}" r="{n(r)}" fill="{fill}"/>'


def particle_shape(p: Particle, fill: str) -> str:
    """A particle of a cell of side 1, drawn in a cell of side SIDE."""
    x, y = p.x * SIDE, p.y * SIDE
    if isinstance(p, Disk):
        return circle(x, y, p.r * SIDE, fill)
    return (f'<ellipse cx="{n(x)}" cy="{n(y)}" rx="{n(p.a * SIDE)}" ry="{n(p.b * SIDE)}" fill="{fill}" '
            f'transform="rotate({n(math.degrees(p.angle))} {n(x)} {n(y)})"/>')


def stroked(path: str, colour: str, width: float) -> str:
    return (f'<path d="{path}" fill="none" stroke="{colour}" stroke-width="{n(width)}" '
            'stroke-linecap="round" stroke-linejoin="round"/>')


def placed(drawing: Drawing, x: float, y: float, height: float) -> str:
    """The drawing scaled to a height, its top left corner at (x, y)."""
    scale = height / drawing.height
    return f'<g transform="translate({n(x)} {n(y)}) scale({scale:.5f})">{drawing.body}</g>'


def svg_document(drawing: Drawing) -> str:
    w, h = n(drawing.width), n(drawing.height)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
            f'role="img" aria-label="GMMD">\n<title>GMMD</title>\n{drawing.body}\n</svg>\n')


# The periodic cell
def read_particles(mic_json: str) -> list[Particle]:
    """The disks and ellipses of a GMMD microstructure file with a cell of side 1."""
    particles = []
    for p in json.loads(mic_json)["particles"]:
        x, y = p["center"]
        if p["shape"] == "Disk":
            particles.append(Disk(x, y, p["r"], p["phase"]))
        else:
            particles.append(Ellipse(x, y, p["major_axis"] / 2, p["minor_axis"] / 2, p["angle"], p["phase"]))
    return particles


def shifted(particles: list[Particle], sx: float, sy: float) -> list[Particle]:
    """The same periodic packing with the origin of the cell moved."""
    return [replace(p, x=(p.x + sx) % 1, y=(p.y + sy) % 1) for p in particles]


def with_images(particles: list[Particle]) -> list[Particle]:
    """The particles and those of their periodic images that reach into the cell."""
    images = [replace(p, x=p.x + i, y=p.y + j) for p in particles for i in (-1, 0, 1) for j in (-1, 0, 1)]
    return [p for p in images
            if -p.half_width < p.x < 1 + p.half_width and -p.half_height < p.y < 1 + p.half_height]


def framing_cost(particles: list[Particle]) -> float:
    """How badly the border of the tile cuts the particles; lower is better.

    A good frame cuts two or three particles near their middle, cuts none at a corner,
    leaves the largest particle whole, and keeps the whole particles clear of the border.
    """
    corner = CORNER / SIDE
    largest = max(p.area for p in particles)
    cost, cut_count = 0.0, 0
    for p in particles:
        # for each side: how far the centre is from it, and how far the particle reaches towards it
        sides = [(p.x, p.half_width), (1 - p.x, p.half_width), (p.y, p.half_height), (1 - p.y, p.half_height)]
        clear = sorted(distance - reach for distance, reach in sides)
        distance, reach = min(sides, key=lambda side: side[0] - side[1])
        if clear[1] < corner:
            cost += 10  # at a corner, where the tile is rounded
        if clear[0] < 0:
            cut_count += 1
            cost += 4 * (distance / reach) ** 2  # a cut far from the middle leaves a sliver
            if p.area == largest:
                cost += 10
        else:
            cost += max(0.0, 0.06 - clear[0]) * 40  # whole but crowding the border
    return cost + 2 * abs(cut_count - 2.5)


def best_framed(particles: list[Particle], steps: int = 48) -> list[Particle]:
    """The packing with the origin of the cell where the tile frames it best."""
    candidates = (shifted(particles, i / steps, j / steps) for i in range(steps) for j in range(steps))
    return min(candidates, key=framing_cost)


# The mark
def phase_colours(theme: Theme) -> dict[str, str]:
    return {"1": theme.ember, "2": theme.paper, "3": theme.teal, "4": theme.teal}


def mark(particles: list[Particle], theme: Theme) -> Drawing:
    """The cell of particles in colour on a rounded tile, cut at its border."""
    colours = phase_colours(theme)
    shapes = "\n".join(particle_shape(p, colours[p.phase]) for p in with_images(particles))
    tile = f'width="{SIDE}" height="{SIDE}" rx="{CORNER}"'
    body = (f'<clipPath id="cell-tile"><rect {tile}/></clipPath>\n'
            f'<rect {tile} fill="{theme.tile}"/>\n'
            f'<g clip-path="url(#cell-tile)">\n{shapes}\n</g>')
    return Drawing(body, SIDE, SIDE)


def one_colour_mark(particles: list[Particle], colour: str, frame: float = 7) -> Drawing:
    """The mark in one colour: the particles are holes in a solid tile.

    The cell sits inside a solid frame, so the particles the cell cuts do not notch the
    outline of the tile.
    """
    tile = f'width="{SIDE}" height="{SIDE}" rx="{CORNER}"'
    inner = SIDE - 2 * frame
    cell = f'x="{n(frame)}" y="{n(frame)}" width="{n(inner)}" height="{n(inner)}" rx="{n(CORNER - frame)}"'
    holes = "\n".join(particle_shape(p, BLACK) for p in with_images(particles))
    body = (f'<clipPath id="cell-inner"><rect {cell}/></clipPath>\n'
            f'<mask id="cell-holes"><rect {tile} fill="{WHITE}"/>\n'
            f'<g clip-path="url(#cell-inner)"><g transform="translate({n(frame)} {n(frame)}) scale({inner / SIDE:.5f})">'
            f'\n{holes}\n</g></g>\n</mask>\n'
            f'<rect {tile} fill="{colour}" mask="url(#cell-holes)"/>')
    return Drawing(body, SIDE, SIDE)


# The name
def letter_g(theme: Theme, weight: float) -> Drawing:
    """A G drawn as a circle with a radius and its centre: a disk and the radius that describes it."""
    c, r = CAP / 2, CAP / 2 - weight / 2
    opening = math.radians(38)
    x0, y0 = c + r * math.cos(opening), c - r * math.sin(opening)
    ring = stroked(f"M{n(x0)} {n(y0)} A{n(r)} {n(r)} 0 1 0 {n(c + r)} {n(c)} H{n(c)}", theme.ink, weight)
    return Drawing(f"{ring}\n{circle(c, c, 0.72 * weight, theme.ember)}", CAP, CAP)


def letter_m(theme: Theme, weight: float) -> Drawing:
    h, width = weight / 2, 0.9 * CAP
    path = f"M{n(h)} {n(CAP - h)} V{n(h)} L{n(width / 2)} {n(0.62 * CAP)} L{n(width - h)} {n(h)} V{n(CAP - h)}"
    return Drawing(stroked(path, theme.ink, weight), width, CAP)


def letter_d(theme: Theme, weight: float) -> Drawing:
    h, r = weight / 2, CAP / 2 - weight / 2
    flat = 0.3 * CAP
    path = f"M{n(h)} {n(h)} H{n(flat)} A{n(r)} {n(r)} 0 0 1 {n(flat)} {n(CAP - h)} H{n(h)}Z"
    return Drawing(stroked(path, theme.ink, weight), flat + r + h, CAP)


def in_a_row(drawings: list[Drawing], gap: float) -> Drawing:
    """Drawings of the same height side by side."""
    parts, x = [], 0.0
    for drawing in drawings:
        parts.append(f'<g transform="translate({n(x)} 0)">{drawing.body}</g>')
        x += drawing.width + gap
    return Drawing("\n".join(parts), x - gap, drawings[0].height)


def name(theme: Theme, weight: float = 17) -> Drawing:
    """GMMD in one line weight."""
    letters = [letter_g(theme, weight), letter_m(theme, weight), letter_m(theme, weight), letter_d(theme, weight)]
    return in_a_row(letters, gap=0.2 * CAP)


def logo(mark_drawing: Drawing, theme: Theme) -> Drawing:
    """The mark with the name at its right."""
    word = name(theme)
    word_height = 0.5 * SIDE
    gap = 0.26 * SIDE
    word_width = word.width * word_height / word.height
    body = f"{mark_drawing.body}\n{placed(word, SIDE + gap, (SIDE - word_height) / 2, word_height)}"
    return Drawing(body, SIDE + gap + word_width, SIDE)


# The files
def logo_files(particles: list[Particle]) -> dict[str, str]:
    """Every file of the logo, as SVG text by file name."""
    versions = [
        ("", mark(particles, ON_LIGHT), ON_LIGHT),
        ("-on-dark", mark(particles, ON_DARK), ON_DARK),
        ("-black", one_colour_mark(particles, BLACK), one_colour(BLACK)),
        ("-white", one_colour_mark(particles, WHITE), one_colour(WHITE)),
    ]
    files = {}
    for suffix, mark_drawing, theme in versions:
        files[f"mark{suffix}.svg"] = svg_document(mark_drawing)
        files[f"logo{suffix}.svg"] = svg_document(logo(mark_drawing, theme))
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mic_json", type=Path, help="microstructure file written by GMMD")
    parser.add_argument("out_dir", type=Path)
    args = parser.parse_args()

    particles = best_framed(read_particles(args.mic_json.read_text()))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for file_name, text in logo_files(particles).items():
        (args.out_dir / file_name).write_text(text)
        print(args.out_dir / file_name)


if __name__ == "__main__":
    main()
