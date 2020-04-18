def plotParticles(particles, dir, grid='off', verlet_ngh=False, center_part=False,
                  show=False, save=False, **kwargs):
    from particle_classes import Particle
    """Plot the particles."""
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    import numpy as np

    N = len(particles)
    if particles[0].dim == 2:
        # Two dimensional problem
        fig = plt.figure()

        ax = plt.gca()

        for i in range(N):
            class_name_i_particle = particles[i].__class__.__name__
            for j in range(-1, 2):
                for k in range(-1, 2):
                    if 'Disk' == class_name_i_particle or 'CylindricalFiber' == class_name_i_particle:
                        circ = mpatches.Circle(
                            particles[i].position_center+np.array([1*j, 1*k]), radius=particles[i].radius, alpha=0.8)
                        ax.add_artist(circ)
                        if verlet_ngh:
                            circ = mpatches.Circle(
                                particles[i].position_center+np.array([1*j, 1*k]+particles[i].displacement_last_verlet), radius=Particle.verlet_factor*particles[i].radius, alpha=0.1)
                            ax.add_artist(circ)
                        if center_part:
                            plt.annotate(xy=particles[i].position_center, s=str(i))
                            plt.scatter(particles[i].position_center[0],
                                        particles[i].position_center[1])
                    if 'Ellipse' == class_name_i_particle:
                        ellip = mpatches.Ellipse(particles[i].position_center+np.array(
                            [1*j, 1*k]), particles[i].major_axis, particles[i].minor_axis,
                            angle=180/np.pi*particles[i].angle, alpha=0.8)
                        ax.add_artist(ellip)
                        if verlet_ngh:
                            ellip = mpatches.Ellipse(particles[i].position_center+np.array([1*j, 1*k]+particles[i].displacement_last_verlet), particles[i].major_axis
                                                     * Particle.verlet_factor, particles[i].minor_axis*Particle.verlet_factor, angle=180/np.pi*particles[i].angle, alpha=0.2)
                            ax.add_artist(ellip)
                        if center_part:
                            plt.annotate(xy=particles[i].position_center, s=str(i))
                            plt.scatter(particles[i].position_center[0],
                                        particles[i].position_center[1], s=0.01)

        if grid == 'cell_list':
            plt.xticks(np.linspace(0, 1, Particle.n_cell_dim+1, endpoint=True))
            plt.yticks(np.linspace(0, 1, Particle.n_cell_dim+1, endpoint=True))
            plt.grid(b=True, which='both')
        elif grid == 'fft':
            discret_spec_array = kwargs('discret_spec_array')
            plt.xticks(np.linspace(
                0, 1, discret_spec_array['rgmsh']['n_voxels_dims'][0]+1, endpoint=True))
            plt.yticks(np.linspace(
                0, 1, discret_spec_array['rgmsh']['n_voxels_dims'][1]+1, endpoint=True))
            plt.grid(b=True, which='both')

        ax.axis("square")

        plt.axis([0, 1, 0, 1])

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

        fig = plt.figure()
        ax = plt.gca()

        is_manual = False  # True if user has taken control of the animation
        interval = 50  # ms, time between animation frames
        loop_len = 5.0  # seconds per loop
        scale = 5

        particle_patches = []
        for i in range(len(particles)):
            for j in range(-1, 2):
                for k in range(-1, 2):
                    if particles[i].__class__.__name__== 'Disk' or 'CylindricalFiber' == particles[i].__class__.__name__:
                        circ = mpatches.Circle(
                            np.array(particles[i].position_center_history)[0, :]+np.array([1*j, 1*k]),
                             radius=particles[i].radius, alpha=0.5)
                        particle_patches.append(ax.add_artist(circ))
                    elif particles[i].__class__.__name__== 'Ellipse':
                        ellip = mpatches.Ellipse(
                            np.array(particles[i].position_center_history)[0, :]+np.array([1*j, 1*k]),
                             particles[i].major_axis, particles[i].minor_axis, angle=180/np.pi*particles[i].angle, alpha=0.5)
                        particle_patches.append(ax.add_artist(ellip))

        ax.axis("square")
        plt.axis([0, 1, 0, 1])

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
            for i in range(len(particles)):
                for j in range(-1, 2):
                    for k in range(-1, 2):
                        particle_patches[9*i + 3*(j+1) + (k+1)].set_center(np.array(particles[i].position_center_history)[int(frame), :]+np.array([1*j, 1*k]))

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

        ani = animation.FuncAnimation(fig, update_plot, len(particles[0].position_center_history)) #, fargs=(is_manual, is_manual), interval=interval)


        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

        ani.save(dp_dir + ".mp4", writer=writer)

        plt.show()


if __name__ == '__main__':
    import os
    import pickle
    import numpy as np
    dir_previous_mic = "/home/zeluis/Documents/Tese/programa/results/to_show/Disk_200_0.6_1"
    for file in os.listdir(dir_previous_mic):
        if file.endswith(".p") and file != 'info_micro.p':
            mic_name = file
    particles = pickle.load(open(dir_previous_mic + '/' + mic_name, 'rb'))
    original_info_dict = pickle.load(open(dir_previous_mic + '/info_micro.p', 'rb'))
    # No need to generate a new microstructure. Using a previous microstructure.
    # reconstructParticleAttributes(particles, rve_dims, original_info_dict)
    # Reconstructing the relevant Particle attributes that could not be pickled
    # createResultsDirectory(particles, dp_dir)
    plotPaths(particles, particles[0].dim, dir_previous_mic)
