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
    if particle.dim == 2:
        radius_x = particle.semi_major_axis
        radius_y = particle.semi_minor_axis
        for i_image in itertools.product(offsets, repeat=2):
            center_x = particle.position_center[0] + rve_dims[0] * i_image[0]
            center_y = particle.position_center[1] + rve_dims[1] * i_image[1]
            if (
                center_x > rve_dims[0] + radius_x
                or center_x < -radius_x
                or center_y > rve_dims[1] + radius_y
                or center_y < -radius_y
            ):
                continue
            yield (center_x, center_y, 0.0)
        # A cylindrical fibre spans the RVE along its own direction, so it has a centre
        # of two coordinates and is enumerated here rather than below
    else:
        radius = particle.radius
        for i_image in itertools.product(offsets, repeat=3):
            center = [
                particle.position_center[i_dir] + rve_dims[i_dir] * i_image[i_dir]
                for i_dir in range(3)
            ]
            if any(
                [
                    center[i_dir] > rve_dims[i_dir] + radius
                    or center[i_dir] < -radius
                    for i_dir in range(3)
                ]
            ):
                continue
            yield tuple(center)
        # The circumscribed radius is used for every shape, so the test discards only
        # images that cannot reach the RVE under any orientation
