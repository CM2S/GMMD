
from particle_classes import Particle

import numpy as np

import scipy.integrate as integrate


def _adjust_bounds(ax, points):
    ptp_bound = points.ptp(axis=0)
    ax.set_xlim(points[:,0].min() - 0.1*ptp_bound[0],
                points[:,0].max() + 0.1*ptp_bound[0])
    ax.set_ylim(points[:,1].min() - 0.1*ptp_bound[1],
                points[:,1].max() + 0.1*ptp_bound[1])



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import matplotlib
    cmap = matplotlib.cm.get_cmap('Blues')
    phi = np.linspace(0, 2*np.pi, 200, endpoint=True)
    color = cmap(phi/(2*np.pi))
    # def normal_dens(phi):
    #     normal_dens = 1 + np.cos(3*phi)
    #     return normal_dens
    # def position_function:
    # radius = integrate.quad(lambda phi: normal_dens*np.exp(1j*phi), 0, phi)
    # radius = ((-1j/30)*(-5 + 30*np.exp((4*1j)*phi) + 3*np.exp((8*1j)*phi)))/np.exp((3*1j)*phi)
    _, axs = plt.subplots(nrows=1, ncols=2)
    norm_dens = np.real(
            1
            + 0.011*(np.exp(2*1j*phi) + np.exp(-2*1j*phi))
            + 0.028*(np.exp(1j*3*phi) + np.exp(-3*1j*phi))
            + 0.945*(np.exp(1j*4*phi) + np.exp(-4*1j*phi))
            + 0.081*(np.exp(1j*5*phi) + np.exp(-5*1j*phi))
            + 0.115*(np.exp(1j*6*phi) + np.exp(-6*1j*phi)))
    plt.sca(axs[0])
    plt.scatter(phi, norm_dens, color=color)
    plt.sca(axs[1])
    radius = 1/(-1j)*np.exp(-1j*phi) \
        + 0.011*(1/1j*np.exp(1j*phi) + 1/(-3*1j)*np.exp(-3*1j*phi)) \
        + 0.028*(1/(2*1j)*np.exp(1j*2*phi) + 1/(-4*1j)*np.exp(-4*1j*phi)) \
        + 0.945*(1/(3*1j)*np.exp(1j*3*phi) + 1/(-5*1j)*np.exp(-5*1j*phi)) \
        + 0.081*(1/(4*1j)*np.exp(1j*4*phi) + 1/(-6*1j)*np.exp(-6*1j*phi)) \
        + 0.115*(1/(5*1j)*np.exp(1j*5*phi) + 1/(-7*1j)*np.exp(-7*1j*phi))
        # 2/(-1j)*np.exp(-1j*phi) + 1/1j*np.exp(1j*phi) + 1/(2*1j)*np.exp(1j*2*phi) + 1/(-3*1j)*np.exp(-3*1j*phi) + 1/(-4*1j)*np.exp(-4*1j*phi)
    plt.axis("equal")
    # plt.plot(phi, 1 + np.cos(3*phi))
    # plt.plot(phi, 3 + 1/2*np.exp(-3*1j*phi) + 1/2*np.exp(3*1j*phi) + 1)
    plt.scatter(np.real(radius), np.imag(radius), color=color)
    plt.show()


    vertices = np.array(
        [[150, 150],
         [150, 350],
         [350, 350],
         [350, 150],
         [250, 167]])
    region = [[4, 3, 2, 1, 0]]
    test_pol = Polygon(vertices, region)

    IMTs = computeIrreducibleMinkowskiTensors(test_pol)[0]
    print(np.abs(IMTs[0]))
    print(np.abs(IMTs[2]/IMTs[0]))
    print(np.abs(IMTs[3]/IMTs[0]))
    print(np.abs(IMTs[4]/IMTs[0]))
    print(np.abs(IMTs[5]/IMTs[0]))
    print(np.abs(IMTs[6]/IMTs[0]))

def set_voronoi_plot_2d(vor, ax=None, **kw):
    """
    Plot the given Voronoi diagram in 2-D

    Parameters
    ----------
    vor : scipy.spatial.Voronoi instance
        Diagram to plot
    ax : matplotlib.axes.Axes instance, optional
        Axes to plot on
    show_points: bool, optional
        Add the Voronoi points to the plot.
    show_vertices : bool, optional
        Add the Voronoi vertices to the plot.
    line_colors : string, optional
        Specifies the line color for polygon boundaries
    line_width : float, optional
        Specifies the line width for polygon boundaries
    line_alpha: float, optional
        Specifies the line alpha for polygon boundaries

    Returns
    -------
    fig : matplotlib.figure.Figure instance
        Figure for the plot

    See Also
    --------
    Voronoi

    Notes
    -----
    Requires Matplotlib.

    """
    from matplotlib.collections import LineCollection
    import matplotlib.pyplot as plt

    if vor.points.shape[1] != 2:
        raise ValueError("Voronoi diagram is not 2-D")

    if kw.get('show_points', True):
        ax.plot(vor.points[:, 0], vor.points[:, 1], '.')
    if kw.get('show_vertices', True):
        ax.plot(vor.vertices[:, 0], vor.vertices[:, 1], 'o')
        # for ind_vert, vert in enumerate(vor.vertices):
        #     plt.text(vert[0], vert[1], str(ind_vert))

    line_colors = kw.get('line_colors', 'k')
    line_width = kw.get('line_width', 1.0)
    line_alpha = kw.get('line_alpha', 1.0)
    
    line_segments = []
    for simplex in vor.ridge_vertices:
        simplex = np.asarray(simplex)
        if np.all(simplex >= 0):
            line_segments.append([(x, y) for x, y in vor.vertices[simplex]])
    
    lc = LineCollection(line_segments,
                        colors=line_colors,
                        lw=line_width,
                        linestyle='solid')
    lc.set_alpha(line_alpha)
    ax.add_collection(lc)
    ptp_bound = vor.points.ptp(axis=0)
    # 
    # line_segments = []
    # center = vor.points.mean(axis=0)
    # for pointidx, simplex in zip(vor.ridge_points, vor.ridge_vertices):
    #     simplex = np.asarray(simplex)
    #     if np.any(simplex < 0):
    #         i = simplex[simplex >= 0][0]  # finite end Voronoi vertex
    # 
    #         t = vor.points[pointidx[1]] - vor.points[pointidx[0]]  # tangent
    #         t /= np.linalg.norm(t)
    #         n = np.array([-t[1], t[0]])  # normal
    # 
    #         midpoint = vor.points[pointidx].mean(axis=0)
    #         direction = np.sign(np.dot(midpoint - center, n)) * n
    #         far_point = vor.vertices[i] + direction * ptp_bound.max()
    # 
    #         line_segments.append([(vor.vertices[i, 0], vor.vertices[i, 1]),
    #                               (far_point[0], far_point[1])])
    # 
    # lc = LineCollection(line_segments,
    #                     colors=line_colors,
    #                     lw=line_width,
    #                     linestyle='dashed')
    # lc.set_alpha(line_alpha)
    # ax.add_collection(lc)
    _adjust_bounds(ax, vor.points)

    return ax.figure


def generateColors(particles):
    """Generate a color for each pahse."""
    colors = {}
    colors_def = ['c', 'r', 'g', 'm', 'b', 'y']
    k_color = 0
    for phase in Particle.list_phases:
    # Running through all the phases
        if phase == Particle.matrix_phase:
            continue
        colors[phase] = colors_def[k_color]
        k_color += 1

    return colors

def plotParticles(particles, dir, grid='off', verlet_ngh=False, center_part=False,
                  show=False, save=False, **kwargs):

    """Plot the particles."""
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    import numpy as np


    N = len(particles)
    if particles[0].dim == 2:
        # Two dimensional problem
        fig = plt.figure()

        ax = plt.gca()

        if Particle.box[0] == Particle.box[1]:
            ax.axis("square")
        else:
            ax.set_aspect('equal', adjustable='box')

        ax.set_ylim(0, Particle.box[1])
        ax.set_xlim(0, Particle.box[0])

        colors = generateColors(particles)

        for i in range(N):
            class_name_i_particle = particles[i].__class__.__name__
            for j in range(-1, 2):
                for k in range(-1, 2):
                    if 'Disk' == class_name_i_particle or 'CylindricalFiber' == class_name_i_particle:
                        circ = mpatches.Circle(
                            particles[i].position_center + Particle.box*np.array([j, k]), radius=particles[i].radius, alpha=0.8, color=colors[particles[i].phase])
                        ax.add_artist(circ)
                        if verlet_ngh:
                            circ = mpatches.Circle(
                                particles[i].position_center+Particle.box*np.array([1*j, 1*k]+particles[i].displacement_last_verlet), radius=Particle.verlet_factor*particles[i].radius, alpha=0.1, color=colors[particles[i].phase])
                            ax.add_artist(circ)
                        if center_part:
                            plt.annotate(xy=particles[i].position_center, s=str(i))
                            plt.scatter(particles[i].position_center[0],
                                        particles[i].position_center[1])
                    if 'Ellipse' == class_name_i_particle:
                        ellip = mpatches.Ellipse(particles[i].position_center+Particle.box*np.array(
                            [1*j, 1*k]), particles[i].major_axis, particles[i].minor_axis,
                            angle=180/np.pi*particles[i].angle, alpha=0.8, color=colors[particles[i].phase])
                        ax.add_artist(ellip)
                        if verlet_ngh:
                            ellip = mpatches.Ellipse(particles[i].position_center+Particle.box*np.array([1*j, 1*k]+particles[i].displacement_last_verlet), particles[i].major_axis
                                                     * Particle.verlet_factor, particles[i].minor_axis*Particle.verlet_factor, angle=180/np.pi*particles[i].angle, alpha=0.2, color=colors[particles[i].phase])
                            ax.add_artist(ellip)
                        if center_part:
                            plt.annotate(xy=particles[i].position_center, s=str(i))
                            plt.scatter(particles[i].position_center[0],
                                        particles[i].position_center[1], s=0.01)

        if grid == 'cell_list':
            plt.xticks(np.linspace(0, 1, Particle.n_cell_dim[0]+1, endpoint=True))
            plt.yticks(np.linspace(0, 1, Particle.n_cell_dim[1]+1, endpoint=True))
            plt.grid(b=True, which='both')
        elif grid == 'fft':
            discret_spec_array = kwargs('discret_spec_array')
            plt.xticks(np.linspace(
                0, 1, discret_spec_array['rgmsh']['n_voxels_dims'][0]+1, endpoint=True))
            plt.yticks(np.linspace(
                0, 1, discret_spec_array['rgmsh']['n_voxels_dims'][1]+1, endpoint=True))
            plt.grid(b=True, which='both')

        
        if save:
            plt.savefig(dir + ".png")

        if show:
            plt.show()

    elif particles[0].dim == 3:
        pass
    else:
        box = Particle.box
        from mpl_toolkits.mplot3d import Axes3D
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
        import matplotlib.pyplot as plt

        def drawSphere(pos, r):
            # draw sphere
            u, v = np.mgrid[0:2*np.pi:5j, 0:np.pi:5j]
            x = np.cos(u)*np.sin(v)
            y = np.sin(u)*np.sin(v)
            z = np.cos(v)
            # shift and scale sphere
            x = r*x + pos[0]
            y = r*y + pos[1]
            z = r*z + pos[2]
            return (x, y, z)

        def plot_cube(cube_definition):
            cube_definition_array = [
                np.array(list(item))
                for item in cube_definition
            ]

            points = []
            points += cube_definition_array
            vectors = [
                cube_definition_array[1] - cube_definition_array[0],
                cube_definition_array[2] - cube_definition_array[0],
                cube_definition_array[3] - cube_definition_array[0]
            ]

            points += [cube_definition_array[0] + vectors[0] + vectors[1]]
            points += [cube_definition_array[0] + vectors[0] + vectors[2]]
            points += [cube_definition_array[0] + vectors[1] + vectors[2]]
            points += [cube_definition_array[0] + vectors[0] + vectors[1] + vectors[2]]

            points = np.array(points)

            edges = [
                [points[0], points[3], points[5], points[1]],
                [points[1], points[5], points[7], points[4]],
                [points[4], points[2], points[6], points[7]],
                [points[2], points[6], points[3], points[0]],
                [points[0], points[2], points[4], points[1]],
                [points[3], points[6], points[7], points[5]]
            ]

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            faces = Poly3DCollection(edges, linewidths=1, edgecolors='k')
            faces.set_facecolor((0, 0, 1, 0.05))

            ax.add_collection3d(faces)

            # Plot the points themselves to force the scaling of the axes
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=0)

            ax.set_aspect('equal')

        cube_definition = [
            (0, 0, 0), (0, 1, 0), (1, 0, 0), (0, 0, 1)
        ]
        plot_cube(cube_definition)

        fig = plt.gcf()
        ax = fig.gca()

        for i in range(N):
            for j in range(-1, 2):
                for k in range(-1, 2):
                    for l in range(-1, 2):
                        (xs, ys, zs) = drawSphere(
                            particles[i].position_center+np.array([1*j, 1*k, 1*l]),
                            particles[i].radius)
                        x_clip = np.logical_or(np.abs(np.array(xs)) > 1, xs < 0)
                        y_clip = np.logical_or(np.abs(np.array(ys)) > 1, ys < 0)
                        z_clip = np.logical_or(np.abs(np.array(zs)) > 1, zs < 0)
                        in_points = np.logical_or(np.logical_or(x_clip, y_clip), z_clip)
                        # xs[in_points] = np.nan
                        # ys[in_points] = np.nan
                        zs[in_points] = np.nan
                        ax.plot_wireframe(xs, ys, zs, color="b")
                        ax.text(particles[i].position_center[0],
                                particles[i].position_center[1],
                                particles[i].position_center[2],
                                str(i))
                        plt.scatter(
                            particles[i].position_center[0],
                            particles[i].position_center[1],
                            particles[i].position_center[2])

        plt.grid(b=False)
        ax.set_aspect('equal')
        ax.set_xlim3d(0, 1)
        ax.set_ylim3d(0, 1)
        ax.set_zlim3d(0, 1)
        ax.set_clip_on(True)
        # plt.axis([0, 1, 0, 1, 0, 1])

    if len(Particle.kinetic_energy_history) > 2:
        fig = plt.figure()
        plt.plot(range(len(Particle.kinetic_energy_history)), Particle.kinetic_energy_history)
        if save:
            plt.savefig(dir + "kinetic_energy" + ".png")

        if show:
            plt.show()

        fig = plt.figure()
        ax = plt.gca()
        for line in Particle.temp_change_steps:
            plt.semilogy([line, line], [np.min(Particle.total_overlap_history), np.max(Particle.total_overlap_history)])
        plt.semilogy([0, len(Particle.total_overlap_history)], [Particle.max_residue, Particle.max_residue])
        plt.ylabel('Relative Energy')
        plt.semilogy(range(len(Particle.total_overlap_history)), Particle.total_overlap_history)
        plt.grid()
        plt.axis([0, len(Particle.total_overlap_history), np.min(Particle.total_overlap_history), np.max(Particle.total_overlap_history)])

        if save:
            plt.savefig(dir + "relative_energy" + ".png")

        if show:
            plt.show()

def plotPaths(particles, dim, dp_dir):
    """Plot particle paths."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
    import matplotlib.animation as animation
    from matplotlib.widgets import Slider
    from mpl_toolkits.mplot3d import Axes3D
    import time
    plt.rcParams['animation.ffmpeg_path'] = "/usr/bin/ffmpeg"



    only_center = False

    if dim == 3:
        fig = plt.figure()
        ax = plt.axes(projection='3d')

        is_manual = False  # True if user has taken control of the animation
        interval = 50  # ms, time between animation frames
        loop_len = 5.0  # seconds per loop
        scale = 5

        k = 0
        x = []
        y = []
        z = []
        for i_particle in particles:
            k += 1
            path_i = np.array(i_particle.position_center_history)
            color = np.random.uniform(size=3)
            c = np.array([np.concatenate((color, np.array([1]))) for i in range(len(path_i[:, 0]))])
            # my_col = map.to_rgba(c)
            x.append(path_i[0, 0])
            y.append(path_i[0, 1])
            z.append(path_i[0, 2])
            ax.scatter(path_i[:, 0], path_i[:, 1], path_i[:, 2], ".", c=c, s=0.01)
        particle_centers = ax.scatter(x, y, z)

        # ax.set_aspect('equal')
        ax.set_xlim3d(0, 1)
        ax.set_ylim3d(0, 1)
        ax.set_zlim3d(0, 1)

        axamp = plt.axes([0.25, .03, 0.50, 0.02])
        # Slider
        samp = Slider(axamp, 'Frame', 0, len(particles[0].position_center_history)-1, valinit=0)

        # Animation controls

        def update_slider(val):
            nonlocal is_manual
            is_manual = True
            update(val)

        def update(frame):
            # update curve
            x = []
            y = []
            z = []
            for i_particle in particles:
                path_i = np.array(i_particle.position_center_history)
                x.append(path_i[int(frame), 0])
                y.append(path_i[int(frame), 1])
                z.append(path_i[int(frame), 2])
            particle_centers._offsets3d = (x, y, z)
            print('new', frame)


        def update_plot_3D(num): #, *args):
            # is_manual = args[0]
            print('num')
            nonlocal is_manual
            if is_manual:
                print('here')
                return particle_centers # don't change

            val = (samp.val + scale) % samp.valmax
            print(val)
            samp.set_val(val)
            is_manual = False  # the above line called update_slider, so we need to reset this
            return particle_centers

        def on_click(event):
            # Check where the click happened
            (xm, ym), (xM, yM)=samp.label.clipbox.get_points()
            if xm < event.x < xM and ym < event.y < yM:
                # Event happened within the slider, ignore since it is handled in update_slider
                return
            else:
                # user clicked somewhere else on canvas = unpause
                nonlocal is_manual
                is_manual = not is_manual

        # call update function on slider value change
        samp.on_changed(update_slider)

        fig.canvas.mpl_connect('button_press_event', on_click)

        ani = animation.FuncAnimation(fig, update_plot_3D, frames=len(particles)-1, blit=False)


        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

        ani.save(dp_dir + ".mp4", writer=writer)

        plt.show()

    elif dim == 2:

        colors = generateColors(particles)

        fig = plt.figure()
        ax = plt.gca()

        if Particle.box[0] == Particle.box[1]:
            ax.axis("square")
        else:
            ax.set_aspect('equal', adjustable='box')

        ax.set_ylim(0, Particle.box[1])
        ax.set_xlim(0, Particle.box[0])


        is_manual = False  # True if user has taken control of the animation
        interval = 50  # ms, time between animation frames
        loop_len = 5.0  # seconds per loop
        scale = 5

        if only_center:
            k = 0
            x = []
            y = []
            for i_particle in particles:
                k += 1
                path_i = np.array(i_particle.position_center_history)
                color = np.random.uniform(size=3)
                c = np.array([np.concatenate((color, np.array([1]))) for i in range(len(path_i[:, 0]))])
                # my_col = map.to_rgba(c)
                x.append(path_i[0, 0])
                y.append(path_i[0, 1])
                plt.scatter(path_i[:, 0], path_i[:, 1], s=0.01, c='black')
            particle_patches = plt.scatter(x, y)
        else:
            particle_patches = []
            for i in range(len(particles)):
                for j in range(-1, 2):
                    for k in range(-1, 2):
                        if particles[i].__class__.__name__== 'Disk' or 'CylindricalFiber' == particles[i].__class__.__name__:
                            circ = mpatches.Circle(
                                np.array(particles[i].position_center_history)[0, :] + Particle.box*np.array([1*j, 1*k]),
                                 radius=particles[i].radius, alpha=0.5, color=colors[particles[i].phase])
                            particle_patches.append(ax.add_artist(circ))
                        elif particles[i].__class__.__name__== 'Ellipse':
                            ellip = mpatches.Ellipse(
                                np.array(particles[i].position_center_history)[0, :]+Particle.box*np.array([1*j, 1*k]),
                                 particles[i].major_axis, particles[i].minor_axis, angle=180/np.pi*particles[i].angle, alpha=0.5, color=colors[particles[i].phase])
                            particle_patches.append(ax.add_artist(ellip))



        axamp = plt.axes([0.25, .03, 0.50, 0.02])
        # Slider
        samp = Slider(axamp, 'Frame', 0, len(particles[0].position_center_history)-1, valinit=0)

        # Animation controls

        def update_slider(val):
            nonlocal is_manual
            is_manual = True
            update(val)

        def update(frame):
            # update curve
            if only_center:
                x = []
                y = []
                for i_particle in particles:
                    path_i = np.array(i_particle.position_center_history)
                    x.append(path_i[int(frame), 0])
                    y.append(path_i[int(frame), 1])
                particle_patches.set_offsets(np.array([x, y]).T)
            else:
                for i in range(len(particles)):
                    for j in range(-1, 2):
                        for k in range(-1, 2):
                            particle_patches[9*i + 3*(j+1) + (k+1)].set_center(np.array(particles[i].position_center_history)[int(frame), :]+Particle.box*np.array([1*j, 1*k]))

        def update_plot(num): #, *args):
            # is_manual = args[0]
            nonlocal is_manual
            if is_manual:
                return particle_patches # don't change

            val = (samp.val + scale) % samp.valmax
            samp.set_val(val)
            is_manual = False  # the above line called update_slider, so we need to reset this
            return particle_patches

        def on_click(event):
            # Check where the click happened
            (xm, ym), (xM, yM)=samp.label.clipbox.get_points()
            if xm < event.x < xM and ym < event.y < yM:
                # Event happened within the slider, ignore since it is handled in update_slider
                return
            else:
                # user clicked somewhere else on canvas = unpause
                nonlocal is_manual
                is_manual = not is_manual

        # call update function on slider value change
        samp.on_changed(update_slider)

        fig.canvas.mpl_connect('button_press_event', on_click)

        ani = animation.FuncAnimation(fig, update_plot, 500) #len(particles[0].position_center_history)) #, fargs=(is_manual, is_manual), interval=interval)


        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

        ani.save(dp_dir + ".mp4", writer=writer)

        plt.show()


def plotPixels(pixel_grid, dir, show=False, save=True):
    import matplotlib.pyplot as plt
    # This import registers the 3D projection, but is otherwise unused.
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
    fig = plt.figure()
    plt.imshow(pixel_grid.T)
    plt.axis([0, np.size(pixel_grid.T, 0), 0, np.size(pixel_grid.T, 1)])
    if save:
        plt.savefig(dir + ".png")
    if show:
        plt.show(block=False)


def plotVoxels(voxel_grid, matrix_phase, list_phase, dir, show=True, save=True):
    import matplotlib.pyplot as plt
    # This import registers the 3D projection, but is otherwise unused.
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    particle_voxels = voxel_grid.T != int(matrix_phase)
    colors = np.empty(particle_voxels.shape, dtype=object)
    color_def = ['c', 'r', 'g', 'y' , 'm', 'b']
    k_color = 0
    for phase in list_phase:
        if phase == matrix_phase:
            continue
        colors[voxel_grid.T == int(phase)] = color_def[k_color]
        k_color += 1
    ax.voxels(particle_voxels, facecolors=colors, edgecolor='k')
    if save:
        plt.savefig(dir + ".png")
    if show:
        plt.show()

def plotVoronoi2D(particles, voronoi, dir, voronoi_type, save=True, show=True):
    """Plot the Voronoi for circular particles."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from scipy.spatial import voronoi_plot_2d

    _ = plt.figure()

    ax = plt.gca()

    N = len(particles)

    if Particle.box[0] == Particle.box[1]:
        ax.axis("square")
    else:
        ax.set_aspect('equal', adjustable='box')

    ax.set_ylim(0, Particle.box[1])
    ax.set_xlim(0, Particle.box[0])

    colors = generateColors(particles)

    for i in range(N):
        class_name_i_particle = particles[i].__class__.__name__
        # particles[i].dilate(global_crit_ero_thick)
        for j in range(-1, 2):
            for k in range(-1, 2):
                if 'Disk' == class_name_i_particle or 'CylindricalFiber' == class_name_i_particle:
                    circ = mpatches.Circle(
                        particles[i].position_center + Particle.box*np.array([j, k]), radius=particles[i].radius, alpha=0.8, color=colors[particles[i].phase])
                    ax.add_artist(circ)
                if 'Ellipse' == class_name_i_particle:
                    ellip = mpatches.Ellipse(particles[i].position_center+Particle.box*np.array(
                        [1*j, 1*k]), particles[i].major_axis, particles[i].minor_axis,
                        angle=180/np.pi*particles[i].angle, alpha=0.8, color=colors[particles[i].phase])
                    ax.add_artist(ellip)

    if voronoi_type == 'set':
        set_voronoi_plot_2d(voronoi, ax=plt.gca(), show_vertices=False)
    elif voronoi_type == 'standard':
        voronoi_plot_2d(voronoi, ax=plt.gca())

    plt.axis([0, Particle.box[0], 0, Particle.box[1]])

    if save:
        plt.savefig(dir + "_voronoi" + ".png")

    if show:
        plt.show()


def plotVoronoi2DwithIMTs(particles, voronoi, IMTs, dir, voronoi_type, save=True, show=True):
    """Plot the Voronoi for circular particles."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib
    from scipy.spatial import voronoi_plot_2d

    for i_order in range(7):

        fig = plt.figure()

        ax = plt.gca()

        N = len(particles)

        if Particle.box[0] == Particle.box[1]:
            ax.axis("square")
        else:
            ax.set_aspect('equal', adjustable='box')

        ax.set_ylim(0, Particle.box[1])
        ax.set_xlim(0, Particle.box[0])

        colors = generateColors(particles)

        for i in range(N):
            class_name_i_particle = particles[i].__class__.__name__
            # particles[i].dilate(global_crit_ero_thick)
            for j in range(-1, 2):
                for k in range(-1, 2):
                    if 'Disk' == class_name_i_particle or 'CylindricalFiber' == class_name_i_particle:
                        circ = mpatches.Circle(
                            particles[i].position_center + Particle.box*np.array([j, k]), radius=particles[i].radius, alpha=0.8, color=colors[particles[i].phase])
                        ax.add_artist(circ)
                    if 'Ellipse' == class_name_i_particle:
                        ellip = mpatches.Ellipse(particles[i].position_center+Particle.box*np.array(
                            [1*j, 1*k]), particles[i].major_axis, particles[i].minor_axis,
                            angle=180/np.pi*particles[i].angle, alpha=0.8, color=colors[particles[i].phase])
                        ax.add_artist(ellip)

        if voronoi_type == 'set':
            set_voronoi_plot_2d(voronoi, ax=plt.gca(), show_vertices=False)
        elif voronoi_type == 'standard':
            voronoi_plot_2d(voronoi, ax=plt.gca())

        plt.axis([0, Particle.box[0], 0, Particle.box[1]])

        cmap = matplotlib.cm.get_cmap('Blues')
        # Initializing the list containing the list of IMTs for each Voronoi cell
        k_cell = 0
        for i_region in voronoi.regions:
            print(i_region)
            if len(i_region) == 0:
                continue
            if any([vertex == -1 for vertex in i_region]):
                print(k_cell)
                continue
        # Running through all the cells in the Voronoi
            # plt.sca(ax)
            if i_order > 0:
                color = cmap(np.abs(IMTs[k_cell][i_order])/np.abs(IMTs[k_cell][0]))
            else:
                color = cmap(np.abs(IMTs[k_cell][0]))
            x = [voronoi.vertices[i_vertex][0] for i_vertex in i_region]
            y = [voronoi.vertices[i_vertex][1] for i_vertex in i_region]
            current_cell = plt.fill(x, y)
            current_cell[0].set_color(color)
            k_cell += 1

        if Particle.box[0] == Particle.box[1]:
            ax.axis("square")
        else:
            ax.set_aspect('equal', adjustable='box')

        ax.set_ylim(0, Particle.box[1])
        ax.set_xlim(0, Particle.box[0])

        colors = generateColors(particles)

        if save:
            plt.savefig(dir + "_" + str(i_order) + ".png")

        if show:
            plt.show()

    for i_order in range(7):

        fig = plt.figure()

        ax = plt.gca()

        N = len(particles)

        plt.hist(np.abs(np.array(IMTs)[:, i_order]), range=(0, 1))

        plt.title(str(i_order))

        if save:
            plt.savefig(dir + "_" + str(i_order) + "_hist" + ".png")

        if show:
            plt.show()
