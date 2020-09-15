import abc


class Thermostat(abc.ABC):

    if thermostat == "multi_temperature":
        # The thermostat used is the isokinetic scheme
        # Setting the options
        if particles[0].dim == 2:
            jump = options.get("equilibration_steps", 25)
            # Number of steps allowed for the system to equilibrate and explore and given
            # temperature before the criterion for temperature lowering is checked
        elif particles[0].dim == 3:
            jump = options.get("equilibration_steps", 25)
        jump_list = []
        last_alt = options.get("inital_temp_steps", 40)
        # Number of steps allowed for the system to equilibrate and explore the initial
        # temperature
        T_ref = options.get("initial_temp", 2.5e10)  # *(particles[0].radius/0.045)**2)
        # Intial temperature
        k_b = 1e-15
        # Analog to the Boltzmann constant
        if kin_energy > 1e-10:
            # Compute the rescaling factor only if the kinetic energy is nonzero
            lambda_vel = np.sqrt(2 * particles[0].dim * N * k_b * T_ref / kin_energy)
            # Rescalling factor (why? 250 -  equipartition theorem)
        else:
            # If the kinetic energy is zero
            lambda_vel = 0
        for i_particle in range(N):
            # Running through all the particles
            particles[i_particle].velocity_center *= lambda_vel
            # Rescalling the velocities
    elif thermostat == "isokinetic":
        T_ref = options.get("initial_temp", 2.5e10)  # *(particles[0].radius/0.045)**2)
        # Intial temperature
        k_b = 1e-15
        # Analog to the Boltzmann constant
        jump = options.get(
            "equilibration_steps", 25
        )  # + 5*100*0.65/(Particle.number*Particle.volume/Particle.volume_RVE))
        # Number of steps allowed for the system to equilibrate and explore and given
        # temperature before the criterion for temperature lowering is checked
        if kin_energy > 1e-10:
            # Compute the rescaling factor only if the kinetic energy is nonzero
            lambda_vel = np.sqrt(2 * particles[0].dim * N * k_b * T_ref / kin_energy)
            # Rescalling factor (why? 250 -  equipartition theorem)
            print("T_ref", T_ref)
        else:
            # If the kinetic energy is zero
            lambda_vel = 0
        for i_particle in range(N):
            # Running through all the particles
            particles[i_particle].velocity_center *= lambda_vel
            # Rescalling the velocities

            if thermostat == "multi_temperature":
                # The thermostat used is the multi_temperature scheme
                if step > last_alt:
                    # If the end of the equilibration time has been reached
                    if Particle.total_overlap > max_residue:
                        # If a legal configuration has not been achieved
                        if any(
                            np.array(Particle.total_overlap_history[-jump // 2 :])
                            - np.array(
                                Particle.total_overlap_history[-jump // 2 - 1 : -1]
                            )
                            > 0
                        ):
                            # If the total overlap has increase in the previous iterations
                            T_ref *= 1 / 4
                            # Lowering the temperature
                            jump += step - last_alt - 1
                            # Updating the equilibration time
                            last_alt = step + jump
                            # Updating the iteration of the last temperature change
                            Particle.temp_change_steps.append(step)
                            jump_list.append(jump)
                            # Saving minimum equilibration times and times at which the
                            # temperature has been lowered
                # Compute the rescaling factor only if the kinetic energy is nonzero
                lambda_vel = np.sqrt(
                    2 * particles[0].dim * N * k_b * T_ref / kin_energy
                )
                # Rescalling factor
                for i_particle in range(N):
                    # Running through all the particles
                    particles[i_particle].velocity_center *= lambda_vel
                    # Rescalling the velocities
                if (
                    relative_energy / Particle.total_overlap < 1e-8
                    and Particle.total_overlap > max_residue
                ):
                    # FIXME: this criterion is giving false positives, relative energy falls
                    # much faster than total overlap
                    pass
            if thermostat == "isokinetic":
                # The thermostate used is the isokinetic with constant temperature
                lambda_vel = np.sqrt(
                    2 * particles[0].dim * N * k_b * T_ref / kin_energy
                )
                for i_particle in range(N):
                    # Running through all the particles
                    particles[i_particle].velocity_center *= lambda_vel
                    # Rescalling the velocities
            else:
                # There is no thermostat
                pass
            if Particle.total_overlap <= max_residue:
                check_tangent = checkTangentToWall(particles, min_distance)
                if check_tangent:
                    # If the configuration has an overlap area smaller than the tolerance
                    n_steps_relax += 1
                    # print('yes',n_steps_relax)
                else:
                    n_steps_relax = 0
                    # Restarting the count
                    forceOutTangentWall(particles, min_distance)
            print_funcs.printToTerminalRefresh(
                step, Particle.total_overlap, relative_energy, kin_energy
            )
            if step > 5 * jump and all(
                (
                    np.abs(
                        np.array(Particle.total_overlap_history[-5 * jump :])
                        - np.array(Particle.total_overlap_history[-5 * jump - 1 : -1])
                    )
                )
                / np.array(Particle.total_overlap_history[-5 * jump - 1 : -1])
                * 100
                < 1e-5
            ):
                print_funcs.printToFile("Failed sample")
                break
