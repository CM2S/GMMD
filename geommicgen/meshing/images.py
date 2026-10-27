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
        zero for a particle that lives in a plane, and for a cylindrical fibre the
        centre of its end, at zero along the direction it runs in.
    """
    offsets = IMAGE_OFFSETS if add_images else (0,)
    radii = (particle.radius,) * particle.dim
    # The circumscribed radius is used for every shape, so the test below discards only
    # images that cannot reach the RVE under any orientation. In the plane the semi
    # axes were taken along x and y whatever the angle of the particle, so an ellipse
    # turned across a face lost the image on the opposite one and the mesh was not
    # periodic. A cylindrical fibre spans the RVE along its own direction, so its
    # centre has two coordinates and it is enumerated in the plane, like a disk
    across = [
        i_dir
        for i_dir in range(len(rve_dims))
        if i_dir != getattr(particle, "direction_fibers", None)
    ]
    # The directions the centre has its coordinates along: every one, except the one a
    # fibre runs in. The images of a fibre along x or y were laid by the first two
    # sides whatever its direction, so in an RVE that is not a cube they were misplaced

    for i_image in itertools.product(offsets, repeat=particle.dim):
        center = [
            particle.position_center[j_ind] + rve_dims[j_dir] * i_image[j_ind]
            for j_ind, j_dir in enumerate(across)
        ]
        if any(
            [
                center[j_ind] > rve_dims[j_dir] + radii[j_ind]
                or center[j_ind] < -radii[j_ind]
                for j_ind, j_dir in enumerate(across)
            ]
        ):
            continue

        point = [0.0, 0.0, 0.0]
        for j_ind, j_dir in enumerate(across):
            point[j_dir] = center[j_ind]
        yield tuple(point)
