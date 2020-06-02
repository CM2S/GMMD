from plotting_functions import plotParticles, plotPaths, plotOverlapHistory, plotKineticEnergyHistory

from particle_classes import Particle

def doMotionAnalysis(particles, rve_dims, dp_dir):
    plotParticles(particles, 0, dp_dir + "_initial_conf", save=True, show=False)
    # Ploting initial configuration
    plotParticles(particles, -1, dp_dir + "_final_config", save=True, show=False)
    # # Ploting final configuration
    # if save_history:
    # plotPaths(particles, particles[0].dim, Particle.file_path)
    for iter, _ in enumerate(particles[0].position_center_history):
        with open(dp_dir + str(iter) + ".vtk", "a") as msh_vtk:
            if particles[0].dim == 2:
                msh_vtk.write("# vtk DataFile Version 2.0")
                msh_vtk.write("\n3D triangulation data")
                msh_vtk.write("\nASCII")
                msh_vtk.write("\n\nDATASET POLYDATA")
                msh_vtk.write("\nPOINTS {0} {1}".format(9*len(particles), 'float'))
                for i_particle in particles:
                    for j in range(-1, 2):
                        for k in range(-1, 2):
                            position = i_particle.position_center_history[iter] + [j, k]*Particle.box
                            msh_vtk.write("\n{0} {1} 0".format(position[0], position[1]))
                msh_vtk.write("\n\nPOINT_DATA {0}".format(9*len(particles)))
                msh_vtk.write("\nSCALARS {0} {1} {2}".format('radius', 'float', '1'))
                msh_vtk.write("\nLOOKUP_TABLE default")
                for i_particle in particles:
                    for j in range(-1, 2):
                        for k in range(-1, 2):
                            msh_vtk.write("\n{0}".format(i_particle.radius))
            elif particles[0].dim == 3:
                msh_vtk.write("# vtk DataFile Version 2.0")
                msh_vtk.write("\n3D triangulation data")
                msh_vtk.write("\nASCII")
                msh_vtk.write("\n\nDATASET POLYDATA")
                msh_vtk.write("\nPOINTS {0} {1}".format(len(particles), 'float'))
                for i_particle in particles:
                    position = i_particle.position_center_history[iter]
                    msh_vtk.write("\n{0} {1} {2}".format(position[0], position[1], position[2]))
                msh_vtk.write("\n\nPOINT_DATA {0}".format(len(particles)))
                msh_vtk.write("\nSCALARS {0} {1} {2}".format('radius', 'float', '1'))
                msh_vtk.write("\nLOOKUP_TABLE default")
                for i_particle in particles:
                    msh_vtk.write("\n{0}".format(i_particle.radius))
    plotOverlapHistory(Particle.total_overlap_history, Particle.temp_change_steps, Particle.max_residue,  dir=dp_dir)
    plotKineticEnergyHistory(Particle.kinetic_energy_history, dir=dp_dir)
