def generateDisks(phase, rve_dims, descriptors):
    """
    Generate disks of *phase* according to *descriptors*.

    Parameters
    ----------
    phase: `.Phase`
        Phase to which the disks will belong.

    rve_dims: list(float)
        List containing the dimensions of the RVE.

    descriptors: dictionary
        Dictionary containing the necesary descriptors to generate the microstructure.
    """
    disks = []
    # Initializing the list containing the disks
    used_parameters = {
        parameter
        for parameter in Disk.possible_parameters
        if any([descriptor.startswith(parameter) for descriptor in descriptors.keys()])
    }
    # Collecting the parameters used
    if any(
        [
            used_parameters == acceptable_description
            for acceptable_description in Disk.acceptable_descriptions
        ]
    ):
        # Checking acceptable sets of parameters
        acceptable_description = True
    else:
        acceptable_description = False
    try:
        if not acceptable_description:
            raise errors.UnacceptableParameters(
                used_parameters, phase.type, Disk.acceptable_descriptions
            )
    except errors.UnacceptableParameters as error:
        error.message()
        quit()
    if "n" in descriptors and "vf" not in descriptors:
        # The desired number of disks was specified
        phase.specNumber(descriptors["n"])
        # Collecting the specified number of particles
        samples = {}
        # Initializing the dictionary containing the samples for each parameter used
        for i_parameter in used_parameters:
            samples[i_parameter] = generateSampleParameter(
                i_parameter,
                Disk.possible_parameters[i_parameter],
                descriptors,
                phase,
                rve_dims,
                n_samples=descriptors["n"],
            )
        r = canonicalParametersDisk(samples, rve_dims)
        for i in range(descriptors["n"]):
            disks.append(Disk(phase, r[i]))
    elif "vf" in descriptors and "n" not in descriptors:
        # The desired volume fraction was specfied
        phase.spec_volume_fraction = descriptors["vf"]
        # Collecting the specified volume fraction
        current_sample = {}
        # Initializing the dictionary containing the samples for each parameter used
        vf_real = 0
        # Initializing the real volume fraction
        while vf_real < descriptors["vf"]:
            for i_parameter in used_parameters:
                current_sample[i_parameter] = generateSampleParameter(
                    i_parameter,
                    Disk.possible_parameters[i_parameter],
                    descriptors,
                    phase,
                    rve_dims,
                )
            r = canonicalParametersDisk(current_sample, rve_dims)
            disks.append(Disk(phase, r[0]))
            vf_real += disks[-1].volume() / (rve_dims[0] * rve_dims[1])
    elif "vf" in descriptors and "n" in descriptors:
        phase.spec_volume_fraction = descriptors["vf"]
        phase.spec_number = descriptors["n"]
        # Collecting the specified volume fraction and number of particles
        samples = {}
        # Initializing the dictionary containing the samples for each parameter used
        for i_parameter in used_parameters:
            samples[i_parameter] = generateSampleParameter(
                i_parameter,
                Disk.possible_parameters[i_parameter],
                descriptors,
                phase,
                rve_dims,
                n_samples=descriptors["n"],
            )
        r = canonicalParametersDisk(samples, rve_dims)
        # Obtaining the radius corresponding to the specified volume fraction and number of
        # particles
        for i in range(descriptors["n"]):
            disks.append(Disk(phase, r))

    return disks


def generateSpheres(phase, rve_dims, descriptors):
    """Generate spheres of *phase* according to *descriptors*.

    Parameters
    ----------
    phase: `.Phase`
        Phase to which the spheres will belong.

    rve_dims: list(float)
        List containing the dimensions of the RVE.

    descriptors: dictionary
        Dictionary containing the necesary descriptors to generate the microstructure.
    """
    spheres = []
    # Initializing the list containing the spheres
    used_parameters = {
        parameter
        for parameter in Sphere.possible_parameters
        if any([descriptor.startswith(parameter) for descriptor in descriptors.keys()])
    }
    # Collecting the parameters used
    if any(
        [
            used_parameters == acceptable_description
            for acceptable_description in Sphere.acceptable_descriptions
        ]
    ):
        acceptable_description = True
    else:
        acceptable_description = False
    # Checking acceptable sets of parameters
    try:
        if not acceptable_description:
            raise errors.UnacceptableParameters(
                used_parameters, phase, Sphere.acceptable_descriptions
            )
    except errors.UnacceptableParameters as error:
        error.message()
        quit()
    if "n" in descriptors and "vf" not in descriptors:
        # The desired number of disks was specified
        phase.specNumber(descriptors["n"])
        # Collecting the specified number of particles
        samples = {}
        # Initializing the dictionary containing the samples for each parameter used
        for i_parameter in used_parameters:
            samples[i_parameter] = generateSampleParameter(
                i_parameter,
                Sphere.possible_parameters[i_parameter],
                descriptors,
                phase,
                rve_dims,
                n_samples=descriptors["n"],
            )
        r = canonicalParametersSphere(samples, rve_dims)
        for i in range(descriptors["n"]):
            spheres.append(Sphere(phase, r[i]))
    elif "vf" in descriptors and "n" not in descriptors:
        # The desired volume fraction was specfied
        phase.spec_volume_fraction = descriptors["vf"]
        # Collecting the specified volume fraction
        current_sample = {}
        # Initializing the dictionary containing the samples for each parameter used
        vf_real = 0
        # Initializing the real volume fraction
        while vf_real < descriptors["vf"]:
            for i_parameter in used_parameters:
                current_sample[i_parameter] = generateSampleParameter(
                    i_parameter,
                    Sphere.possible_parameters[i_parameter],
                    descriptors,
                    phase,
                    rve_dims,
                )
            r = canonicalParametersSphere(current_sample, rve_dims)
            spheres.append(Sphere(phase, r))
            vf_real += spheres[-1].volume() / (rve_dims[0] * rve_dims[1] * rve_dims[2])
    elif "vf" in descriptors and "n" in descriptors:
        phase.spec_volume_fraction = descriptors["vf"]
        phase.spec_number = descriptors["n"]
        # Collecting the specified volume fraction and number of particles
        samples = {}
        # Initializing the dictionary containing the samples for each parameter used
        for i_parameter in used_parameters:
            samples[i_parameter] = generateSampleParameter(
                i_parameter,
                Sphere.possible_parameters[i_parameter],
                descriptors,
                phase,
                rve_dims,
                n_samples=descriptors["n"],
            )
        r = canonicalParametersSphere(samples, rve_dims)
        for i in range(descriptors["n"]):
            spheres.append(Sphere(phase, r))

    return spheres


def generateEllipses(phase, rve_dims, descriptors):
    """Generate ellipses of *phase* according to *descriptors*.

    Parameters
    ----------
    phase: `.Phase`
        Phase to which the spheres will belong.

    rve_dims: list(float)
        List containing the dimensions of the RVE.

    descriptors: dictionary
        Dictionary containing the necesary descriptors to generate the microstructure.
    """
    ellipses = []
    # Initializing the list containing the disks
    used_parameters = {
        parameter
        for parameter in Ellipse.possible_parameters
        if any([descriptor.startswith(parameter) for descriptor in descriptors.keys()])
    }
    # Collecting the parameters used
    if any(
        [
            used_parameters == acceptable_description
            for acceptable_description in Ellipse.acceptable_descriptions
        ]
    ):
        acceptable_description = True
    else:
        acceptable_description = False
    # Checking acceptable sets of parameters
    try:
        if not acceptable_description:
            raise errors.UnacceptableParameters(
                used_parameters, phase, Ellipse.acceptable_descriptions
            )
    except errors.UnacceptableParameters as error:
        error.message()
        quit()
    if "n" in descriptors and "vf" not in descriptors:
        # The desired number of ellipses was specified
        phase.specNumber(descriptors["n"])
        # Collecting the specified number of particles
        samples = {}
        # Initializing the dictionary containing the samples for each parameter used
        for i_parameter in used_parameters:
            samples[i_parameter] = generateSampleParameter(
                i_parameter,
                Ellipse.possible_parameters[i_parameter],
                descriptors,
                phase,
                rve_dims,
                n_samples=descriptors["n"],
            )
        [major_axis, minor_axis, angle] = canonicalParametersEllipse(samples, rve_dims)
        for i in range(descriptors["n"]):
            ellipses.append(Ellipse(phase, major_axis[i], minor_axis[i], angle[i]))
    elif "vf" in descriptors and "n" not in descriptors:
        # The desired volume fraction was specfied
        phase.spec_volume_fraction = descriptors["vf"]
        # Collecting the specified volume fraction
        current_sample = {}
        # Initializing the dictionary containing the samples for each parameter used
        vf_real = 0
        # Initializing the real volume fraction
        while vf_real < descriptors["vf"]:
            for i_parameter in used_parameters:
                current_sample[i_parameter] = generateSampleParameter(
                    i_parameter,
                    Ellipse.possible_parameters[i_parameter],
                    descriptors,
                    phase,
                    rve_dims,
                )
            [major_axis, minor_axis, angle] = canonicalParametersEllipse(
                current_sample, rve_dims
            )
            ellipses.append(Ellipse(phase, major_axis, minor_axis, angle))
            vf_real += ellipses[-1].volume() / (rve_dims[0] * rve_dims[1])
    elif "vf" in descriptors and "n" in descriptors:
        phase.spec_volume_fraction = descriptors["vf"]
        phase.spec_number = descriptors["n"]
        # Collecting the specified volume fraction and number of particles
        samples = {}
        # Initializing the dictionary containing the samples for each parameter used
        for i_parameter in used_parameters:
            samples[i_parameter] = generateSampleParameter(
                i_parameter,
                Ellipse.possible_parameters[i_parameter],
                descriptors,
                phase,
                rve_dims,
                n_samples=descriptors["n"],
            )
        [major_axis, minor_axis, angle] = canonicalParametersEllipse(samples, rve_dims)
        for i in range(descriptors["n"]):
            ellipses.append(Ellipse(phase, major_axis[i], minor_axis[i], angle[i]))

    return ellipses


def generateEllipsoids(phase, rve_dims, descriptors):
    """Generate ellipsoids belonging to *phase* characterized by *descriptors*.

    Parameters
    ----------
    phase: `.Phase`
        Phase to which the spheres will belong.

    rve_dims: list(float)
        List containing the dimensions of the RVE.

    descriptors: dictionary
        Dictionary containing the necesary descriptors to generate the microstructure.
    """
    ellipsoids = []
    # Initializing the list containing the disks
    used_parameters = {
        parameter
        for parameter in Ellipsoid.possible_parameters
        if any([descriptor.startswith(parameter) for descriptor in descriptors.keys()])
    }
    # Collecting the parameters used
    if any(
        [
            used_parameters == acceptable_description
            for acceptable_description in Ellipsoid.acceptable_descriptions
        ]
    ):
        acceptable_description = True
    else:
        acceptable_description = False
    # Checking acceptable sets of parameters
    try:
        if not acceptable_description:
            raise errors.UnacceptableParameters(
                used_parameters, phase, Ellipsoid.acceptable_descriptions
            )
    except errors.UnacceptableParameters as error:
        error.message()
        quit()
    if "n" in descriptors and "vf" not in descriptors:
        # The desired number of ellipsoids was specified
        phase.specNumber(descriptors["n"])
        # Collecting the specified number of particles
        samples = {}
        # Initializing the dictionary containing the samples for each parameter used
        for i_parameter in used_parameters:
            samples[i_parameter] = generateSampleParameter(
                i_parameter,
                Ellipsoid.possible_parameters[i_parameter],
                descriptors,
                phase,
                rve_dims,
                n_samples=descriptors["n"],
            )
        [
            axis_1,
            axis_2,
            axis_3,
            rot_axis_comp_x,
            rot_axis_comp_y,
            rot_axis_comp_z,
            angle,
        ] = canonicalParametersEllipsoid(samples, rve_dims)
        for i in range(descriptors["n"]):
            ellipsoids.append(
                Ellipsoid(
                    phase,
                    axis_1[i],
                    axis_2[i],
                    axis_3[i],
                    rot_axis_comp_x[i],
                    rot_axis_comp_y[i],
                    rot_axis_comp_z[i],
                    angle[i],
                )
            )
    elif "vf" in descriptors and "n" not in descriptors:
        # The desired volume fraction was specfied
        phase.spec_volume_fraction = descriptors["vf"]
        # Collecting the specified volume fraction
        current_sample = {}
        # Initializing the dictionary containing the samples for each parameter used
        vf_real = 0
        # Initializing the real volume fraction
        while vf_real < descriptors["vf"]:
            for i_parameter in used_parameters:
                current_sample[i_parameter] = generateSampleParameter(
                    i_parameter,
                    Ellipsoid.possible_parameters[i_parameter],
                    descriptors,
                    phase,
                    rve_dims,
                )
            [
                axis_1,
                axis_2,
                axis_3,
                rot_axis_comp_x,
                rot_axis_comp_y,
                rot_axis_comp_z,
                angle,
            ] = canonicalParametersEllipsoid(current_sample, rve_dims)
            ellipsoids.append(
                Ellipsoid(
                    phase,
                    axis_1[0],
                    axis_2[0],
                    axis_3[0],
                    rot_axis_comp_x[0],
                    rot_axis_comp_y[0],
                    rot_axis_comp_z[0],
                    angle[0],
                )
            )
            vf_real += ellipsoids[-1].volume() / (rve_dims[0] * rve_dims[1])
    elif "vf" in descriptors and "n" in descriptors:
        phase.spec_volume_fraction = descriptors["vf"]
        phase.spec_number = descriptors["n"]
        # Collecting the specified volume fraction and number of particles
        samples = {}
        # Initializing the dictionary containing the samples for each parameter used
        for i_parameter in used_parameters:
            samples[i_parameter] = generateSampleParameter(
                i_parameter,
                Ellipsoid.possible_parameters[i_parameter],
                descriptors,
                phase,
                rve_dims,
                n_samples=descriptors["n"],
            )
        [
            axis_1,
            axis_2,
            axis_3,
            rot_axis_comp_x,
            rot_axis_comp_y,
            rot_axis_comp_z,
            angle,
        ] = canonicalParametersEllipsoid(samples, rve_dims)
        for i in range(descriptors["n"]):
            ellipsoids.append(
                Ellipsoid(
                    phase,
                    axis_1[i],
                    axis_2[i],
                    axis_3[i],
                    rot_axis_comp_x[i],
                    rot_axis_comp_y[i],
                    rot_axis_comp_z[i],
                    angle[i],
                )
            )

    return ellipsoids


def generateCylindricalFibers(phase, rve_dims, descriptors):
    """
    Generate cylindrical fibers of *phase* according to *descriptors*.

    Parameters
    ----------
    phase: `.Phase`
        Phase to which the fibers will belong.

    rve_dims: list(float)
        List containing the dimensions of the RVE.

    descriptors: dictionary
        Dictionary containing the necesary descriptors to generate the microstructure.
    """
    fibers = []
    # Initializing the list containing the fibers
    used_parameters = {
        parameter
        for parameter in CylindricalFiber.possible_parameters
        if any([descriptor.startswith(parameter) for descriptor in descriptors.keys()])
    }
    # Collecting the parameters used
    if any(
        [
            used_parameters == acceptable_description
            for acceptable_description in CylindricalFiber.acceptable_descriptions
        ]
    ):
        acceptable_description = True
    else:
        acceptable_description = False
    # Checking acceptable sets of parameters
    try:
        if not acceptable_description:
            raise errors.UnacceptableParameters(
                used_parameters, phase, CylindricalFiber.acceptable_descriptions
            )
    except errors.UnacceptableParameters as error:
        error.message()
        quit()
    if "n" in descriptors and "vf" not in descriptors:
        # The desired number of fibers was specified
        phase.specNumber(descriptors["n"])
        # Collecting the specified number of particles
        samples = {}
        # Initializing the dictionary containing the samples for each parameter used
        for i_parameter in used_parameters:
            samples[i_parameter] = generateSampleParameter(
                i_parameter,
                CylindricalFiber.possible_parameter[i_parameter],
                descriptors,
                phase,
                rve_dims,
                n_samples=descriptors["n"],
            )
        r = canonicalParametersDisk(samples, rve_dims)
        for i in range(descriptors["n"]):
            fibers.append(
                CylindricalFiber(phase, r[i], descriptors["direction"], rve_dims)
            )
    elif "vf" in descriptors and "n" not in descriptors:
        # The desired volume fraction was specfied
        phase.spec_volume_fraction = descriptors["vf"]
        # Collecting the specified volume fraction
        current_sample = {}
        # Initializing the dictionary containing the samples for each parameter used
        vf_real = 0
        # Initializing the real volume fraction
        while vf_real < descriptors["vf"]:
            for i_parameter in used_parameters:
                current_sample[i_parameter] = generateSampleParameter(
                    i_parameter,
                    CylindricalFiber.possible_parameter[i_parameter],
                    descriptors,
                    phase,
                    rve_dims,
                )
            r = canonicalParametersDisk(current_sample, rve_dims)
            fibers.append(
                CylindricalFiber(phase, r, descriptors["direction"], rve_dims)
            )
            vf_real += fibers[-1].volume() / (rve_dims[0] * rve_dims[1])
    elif "vf" in descriptors and "n" in descriptors:
        phase.spec_volume_fraction = descriptors["vf"]
        phase.spec_number = descriptors["n"]
        # Collecting the specified volume fraction and number of particles
        samples = {}
        # Initializing the dictionary containing the samples for each parameter used
        for i_parameter in used_parameters:
            samples[i_parameter] = generateSampleParameter(
                i_parameter,
                Ellipsoid.possible_parameters[i_parameter],
                descriptors,
                phase,
                rve_dims,
                n_samples=descriptors["n"],
            )
        r = canonicalParametersDisk(samples, rve_dims)
        for i in range(descriptors["n"]):
            fibers.append(
                CylindricalFiber(phase, r, descriptors["direction"], rve_dims)
            )

    return fibers


def generateSampleParameter(
    parameter, parameter_name, descriptors, phase, rve_dims, n_samples=1, max_sample=100
):
    """Generate a sample of values for *parameter* according to descriptors"""
    size_geom_param = {"r", "major_axis", "minor_axis", "axis_1", "axis_2", "axis_3"}
    # Geometrical parameters related to the size of the particle that must larger than
    # ans smaller than half the size of the smallest dimension of the RVE
    if descriptors.get(parameter + "_distribution") == "uniform":
        # the radius follows a uniform distribution
        try:
            if parameter + "_low" not in descriptors:
                # Checking if the  lower bound was supplied
                raise errors.ParameterMissing(parameter + "_low", phase.type)
            elif parameter + "_high" not in descriptors:
                # Checking if the upper bound was supplied
                raise errors.ParameterMissing(parameter + "_high", phase.type)
            elif descriptors[parameter + "_low"] >= descriptors[parameter + "_high"]:
                # Checking if the lower bound is smaller than the upper bound
                raise errors.UnexpectedValue(
                    descriptors[parameter + "_low"],
                    "{0}_low of phase {1}".format(parameter, phase.type),
                    "smaller than "
                    + parameter
                    + "_high: {0}".format(descriptors[parameter + "_high"]),
                )
        except (errors.ParameterMissing, errors.UnexpectedValue) as error:
            # One of the parameters is missing
            error.message()
            quit()
            # Printing message and aborting
        try:
            if parameter in size_geom_param:
                # Checking if the parameter is a size parameter
                if descriptors[parameter + "_low"] < 0:
                    # Ensuring that it will not produce values smaller than 0
                    raise errors.UnexpectedValue(
                        descriptors[parameter + "_low"],
                        "{0}_low of phase {1}".format(parameter, phase),
                        "larger than 0",
                    )
                elif descriptors[parameter + "_high"] > np.min(rve_dims) / 2:
                    # Ensuring that it will not produce values larger than half the size of the
                    # smallest dimension of the RVE
                    raise errors.UnexpectedValue(
                        descriptors[parameter + "_high"],
                        "{0}_high of phase {1}".format(parameter, phase),
                        "smaller than half the smallest dimension of the RVE: {0}".format(
                            np.min(rve_dims) / 2
                        ),
                    )
        except errors.UnexpectedValue as error:
            error.message()
            quit()
        phase.addGeomParameter(
            parameter_name,
            "Uniform",
            [
                ("Lower bound", descriptors[parameter + "_low"]),
                ("Upper bound", descriptors[parameter + "_high"]),
            ],
        )
        sample = np.random.uniform(
            low=descriptors[parameter + "_low"],
            high=descriptors[parameter + "_high"],
            size=n_samples,
        )
    elif descriptors.get(parameter + "_distribution") == "normal":
        # the radius follows a normal distribution: the paramaters 'r_sigma', the standard
        # deviation of the distribution and 'r_mean', the mean of the distribution, are needed
        try:
            if parameter + "_sigma" not in descriptors:
                raise errors.ParameterMissing(parameter + "_sigma", phase)
            elif parameter + "_mean" not in descriptors:
                raise errors.ParameterMissing(parameter + "_mean", phase)
        except errors.ParameterMissing as error:
            # One of the parameters is missing
            error.message()
            quit()
            # Printing message and aborting
        try:
            if parameter in size_geom_param:
                # Geometric size parameters
                low_25_prob = (
                    descriptors[parameter + "_mean"]
                    - 2 * descriptors[parameter + "_sigma"]
                )
                # Upper bound of tail with 2.5% probability
                high_25_prob = (
                    descriptors[parameter + "_mean"]
                    + 2 * descriptors[parameter + "_sigma"]
                )
                # Lower bound of tail with 2.5% probability
                if low_25_prob < 0:
                    # Ensuring that the probability of generating a value smaller than 0 is
                    # not greater than 2.5%
                    raise errors.DangerousValueNormal(parameter, phase, "low")
                elif high_25_prob > np.min(rve_dims) / 2:
                    # Ensuring that the probability of generating a value larger than half the
                    # size of the smallest dimension of the RVE is not greater than 2.5%
                    raise errors.DangerousValueNormal(parameter, phase, "high")
        except errors.DangerousValueNormal as error:
            error.message()
            quit()
        k_sample = 0
        acceptable_values = False
        while k_sample < max_sample and not acceptable_values:
            phase.addGeomParameter(
                parameter_name,
                "Normal",
                [
                    ("Mean", descriptors[parameter + "_mean"]),
                    ("Std Var", descriptors[parameter + "_sigma"]),
                ],
            )
            sample = np.random.normal(
                loc=descriptors[parameter + "_mean"],
                scale=descriptors[parameter + "_sigma"],
                size=n_samples,
            )
            # Generate a sample
            if parameter in size_geom_param:
                # Geometric size parameters
                if all(
                    [
                        (i_sample > 0 and i_sample <= np.min(rve_dims) / 2)
                        for i_sample in sample
                    ]
                ):
                    # All the values for the geometric size parameters are acceptable
                    acceptable_values = True
            else:
                # Other parameters
                acceptable_values = True
                # Any sample is acceptable
            k_sample += 1
        try:
            if not acceptable_values:
                # No acceptable sample was generated
                raise errors.UnableToGenerateSample(parameter, phase, max_sample)
        except errors.UnableToGenerateSample as error:
            error.message()
            quit()
    elif descriptors.get(parameter + "_distribution") == "discrete":
        # the radius follows a discrete distribution
        values = []
        probabilities = []
        # Initializing the lists containing the values and corresponding probabilities that
        # characterize the discrete distribution required
        for i_descriptor in descriptors:
            if i_descriptor.startswith(parameter + "_value"):
                try:
                    if parameter in size_geom_param:
                        # Checking if the parameter is a size parameter
                        if descriptors[i_descriptor] < 0:
                            # Ensuring that it will not produce values smaller than 0
                            raise errors.UnexpectedValue(
                                descriptors[i_descriptor],
                                "{0} of phase {1}".format(i_descriptor, phase),
                                "larger than 0",
                            )
                        elif descriptors[i_descriptor] > np.min(rve_dims) / 2:
                            # Ensuring that it will not produce values larger than half the size
                            # of the smallest dimension of the RVE
                            raise errors.UnexpectedValue(
                                descriptors[i_descriptor],
                                "{0} of phase {1}".format(i_descriptor, phase),
                                "smaller than half the smallest dimension"
                                + "of the RVE: {0}".format(np.min(rve_dims) / 2),
                            )
                except errors.UnexpectedValue as error:
                    error.message()
                    quit()
                values.append(descriptors[i_descriptor])
                # Save the value
                try:
                    if parameter + "_prob_" + i_descriptor[-1] not in descriptors:
                        raise errors.ParameterMissing(
                            parameter + "_prob_" + i_descriptor[-1], phase
                        )
                    probabilities.append(
                        descriptors[parameter + "_prob_" + i_descriptor[-1]]
                    )
                except errors.ParameterMissing as error:
                    error.message()
                    quit()
        if len(values) == 0:
            # There are no values for the radius parameter
            try:
                raise errors.ParameterMissing(parameter + "value_1", phase)
            except errors.ParameterMissing as error:
                error.message()
                quit()
        elif np.abs(np.sum(probabilities) - 1) > 0.01:
            # The probabilities do not add up to 100%
            try:
                raise errors.ParameterErrorDiscreteProb("r_1", phase)
            except errors.ParameterMissing as error:
                error.message()
                quit()
        value_prob_pairs = [
            [
                ("Value {0}".format(ind + 1), val),
                ("Proability {0}".format(ind + 1), prob),
            ]
            for (ind, val), prob in zip(enumerate(values), probabilities)
        ]
        value_prob_pairs_flat = [
            item for sublist in value_prob_pairs for item in sublist
        ]
        phase.addGeomParameter(parameter_name, "Discrete", value_prob_pairs_flat)
        sample = np.random.choice(values, n_samples, p=probabilities)
    elif parameter + "_distribution" in descriptors:
        # A distribution was specified but it is not supported
        try:
            raise errors.UnsupportedDistribution(
                descriptors[parameter + "_distribution"], parameter, phase
            )
        except errors.UnsupportedDistribution as error:
            error.message()
            quit()
    else:
        # A single value was specified
        try:
            if parameter not in descriptors:
                raise errors.ParameterMissing(parameter, phase)
        except errors.ParameterMissing as error:
            error.message()
            quit()
        try:
            if parameter in size_geom_param:
                # Checking if the parameter is a size parameter
                if descriptors[parameter] < 0:
                    # Ensuring that it will not produce values smaller than 0
                    raise errors.UnexpectedValue(
                        descriptors[parameter],
                        "{0} of phase {1}".format(parameter, phase),
                        "larger than 0",
                    )
                elif descriptors[parameter] > np.min(rve_dims) / 2:
                    # Ensuring that it will not produce values larger than half the size of the
                    # smallest dimension of the RVE
                    raise errors.UnexpectedValue(
                        descriptors[parameter],
                        "{0} of phase {1}".format(parameter, phase),
                        "smaller than half the smallest dimension of the RVE: {0}".format(
                            np.min(rve_dims) / 2
                        ),
                    )
        except errors.UnexpectedValue as error:
            error.message()
            quit()
        if parameter != "n" and parameter != "vf":
            phase.addGeomParameter(
                parameter_name, "Fixed", ("Value", descriptors[parameter])
            )
        sample = np.full((n_samples), descriptors[parameter])

    return sample


def canonicalParametersEllipse(sample, rve_dims):
    """Convert the paramters in *sample* to *major_axis*, *minor_axis* and *angle*."""
    if "major_axis" in sample and "minor_axis" in sample:
        # Both major and minor axis were supplied
        major_axis = np.max([sample["major_axis"], sample["minor_axis"]], axis=0)
        minor_axis = np.min([sample["major_axis"], sample["minor_axis"]], axis=0)
        # Ensuring that the major axis is greater than the minor axis
    elif "major_axis" in sample and "vf" in sample and "n" in sample:
        # The major_axis, the volume faction and the number of particles were supplied
        volume_part = sample["vf"][0] * rve_dims[0] * rve_dims[1] / sample["n"][0]
        aux_minor_axis = volume_part / (np.pi * sample["major_axis"] * 1 / 4)
        # Minor axis computed assuming that all particles have the same area
        major_axis = np.max([sample["major_axis"], aux_minor_axis], axis=0)
        minor_axis = np.min([sample["major_axis"], aux_minor_axis], axis=0)
        # Ensuring that the major axis is greater than the minor axis
    # FIXME: Warnign that all particles will have the same volume
    elif "minor_axis" in sample and "vf" in sample and "n" in sample:
        # The minor axis, the volume faction and the number of particles were supplied
        volume_part = sample["vf"][0] * rve_dims[0] * rve_dims[1] / sample["n"][0]
        aux_major_axis = volume_part / (np.pi * sample["minor_axis"] * 1 / 4)
        # Minor axis computed assuming that all particles have the same area
        major_axis = np.max([aux_major_axis, sample["minor_axis"]], axis=0)
        minor_axis = np.min([aux_major_axis, sample["minor_axis"]], axis=0)
        # Ensuring that the major axis is greater than the minor axis
    elif "ratio" in sample and "vf" in sample and "n" in sample:
        volume_part = sample["vf"][0] * rve_dims[0] * rve_dims[1] / sample["n"][0]
        minor_axis = np.sqrt(volume_part / (np.pi * sample["ratio"] * 1 / 4))
        major_axis = sample["ratio"] * minor_axis
    if "angle" in sample:
        angle = sample["angle"]

    return [major_axis, minor_axis, angle]


def canonicalParametersDisk(sample, rve_dims):
    """Convert the paramters in *sample* to *r*."""
    if "r" in sample:
        # The radius was supplied
        r = sample["r"]
    elif "area" in sample:
        # The area of each particle was supplied
        r = np.sqrt(sample["area"] / np.pi)
    elif "vf" in sample and "n" in sample:
        # Both the volume fraction and the number of particles was supplied
        area = sample["vf"][0] * rve_dims[0] * rve_dims[1] / sample["n"][0]
        # Area of each particle (all the same)
        r = np.sqrt(area / np.pi)
    return r


def canonicalParametersSphere(sample, rve_dims):
    """Convert the parameters in *sample* to *r* characterizing a sphere."""
    if "r" in sample:
        # The radius was supplied
        r = sample["r"]
    elif "volume" in sample:
        # The area of each particle was supplied
        r = np.cbrt(sample["volume"] / (4 / 3 * np.pi))
    elif "vf" in sample and "n" in sample:
        # Both the volume fraction and the number of particles was supplied
        volume = (
            sample["vf"][0] * rve_dims[0] * rve_dims[1] * rve_dims[2] / sample["n"][0]
        )
        # Area of each particle (all the same)
        r = np.cbrt(volume / (4 / 3 * np.pi))
    return r


def canonicalParametersEllipsoid(sample, rve_dims):
    """Convert parameters in *sample* to canonical params characterizing an Ellipsoid."""
    if "axis_1" in sample and "axis_2" in sample and "axis_3":
        # All axis were supplied
        axis_1 = sample["axis_1"]
        axis_2 = sample["axis_2"]
        axis_3 = sample["axis_3"]
    if (
        "ratio_12" in sample
        and "ratio_13" in sample
        and "vf" in sample
        and "n" in sample
    ):
        volume = sample["vf"] * rve_dims[0] * rve_dims[1] * rve_dims[2] / sample["n"]
        axis_1 = np.cbrt(
            volume * sample["ratio_12"] * sample["ratio_13"] * 8 / (np.pi * 4 / 3)
        )
        axis_2 = axis_1 / sample["ratio_12"]
        axis_3 = axis_1 / sample["ratio_13"]
        print(
            "axis",
            axis_1,
            axis_2,
            axis_3,
            "volume",
            sample["n"] * 4 / 3 * axis_1 * axis_2 * axis_3 / 8 * np.pi,
            sample["vf"],
            sample["n"],
        )
    if "angle" in sample:
        angle = sample["angle"]
    if (
        "rot_axis_comp_x" in sample
        and "rot_axis_comp_y" in sample
        and "rot_axis_comp_z" in sample
    ):
        # Euler angles
        rot_axis_comp_x = sample["rot_axis_comp_x"]
        rot_axis_comp_y = sample["rot_axis_comp_y"]
        rot_axis_comp_z = sample["rot_axis_comp_z"]

    return [
        axis_1,
        axis_2,
        axis_3,
        rot_axis_comp_x,
        rot_axis_comp_y,
        rot_axis_comp_z,
        angle,
    ]
