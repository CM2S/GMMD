"""
Module containing the periodic images of a particle.

A particle whose body crosses a face of the RVE has to appear on the opposite face as
well, or the geometry handed to a mesher is not periodic. The images are the copies of
the particle translated by whole RVE dimensions; only the ones that actually reach the
RVE are of any use, and the rest are discarded here rather than by the mesher.
"""

import itertools

IMAGE_OFFSETS = (-1, 0, 1)
# Translations, in RVE dimensions, that a particle crossing one face can need


def periodic_images(particle, rve_dims, add_images=True):
    """
    Give the centres of a particle and of the periodic images that reach the RVE.

    Parameters
    ----------
    particle: `.Particle`
        Particle whose images are wanted.

    rve_dims: list(float)
        Dimensions of the microstructure in each spatial direction.

    add_images: bool
        Whether the periodic images are wanted at all. When False only the particle
        itself is given, which is what a view of the microstructure needs.

    Yields
    ------
    tuple
        Coordinates *(x, y, z)* of the centre of one image, with a third coordinate of
        zero for a particle that lives in a plane.
    """
    offsets = IMAGE_OFFSETS if add_images else (0,)
    radii = (particle.radius,) * particle.dim
    # The circumscribed radius is used for every shape, so the test below discards only
    # images that cannot reach the RVE under any orientation. In the plane the semi
    # axes were taken along x and y whatever the angle of the particle, so an ellipse
    # turned across a face lost the image on the opposite one and the mesh was not
    # periodic. A cylindrical fibre spans the RVE along its own direction, so its
    # centre has two coordinates and it is enumerated in the plane, like a disk

    for i_image in itertools.product(offsets, repeat=particle.dim):
        center = [
            particle.position_center[i_dir] + rve_dims[i_dir] * i_image[i_dir]
            for i_dir in range(particle.dim)
        ]
        if any(
            [
                center[i_dir] > rve_dims[i_dir] + radii[i_dir]
                or center[i_dir] < -radii[i_dir]
                for i_dir in range(particle.dim)
            ]
        ):
            continue

        yield tuple(center) + (0.0,) * (3 - particle.dim)
