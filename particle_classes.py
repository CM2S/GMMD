import numpy as np

from scipy import integrate

import error_classes as errors

import os


class RVE():
    """
    This is the class for an RVE. Contains all the information related to the RVE for use
    in later analysis, while the class attributes are used during the simulation.

    Attributes
    ----------
    particles: `.Particle`
        Particles inside the RVE.

    rve_dims: list(float)
        List containing the dimensions of the RVE.

    dim: float
        Dimension of the RVE.

    box: list(float)
        Dimensions of the simulation box. Note: not necessarily equal to the RVE (e.g.
        cylindrical fibers).

    volume_particles: float
        Total volume of the particles in the RVE.

    number_particles: integer
        Number of particles in the RVE.

    volume_phase: list(float)
        List containing the volume of the particles in each phase.

    volume_RVE: float
        Volume of the RVE.

    relative_energy_history: list(float)
        List containing the relative energy at each iteration.

    kinetic_energy_history: list(float)
        List containing the kinetic energy at each iteration.

    list_phase: list(str)
        List containing the phases in the RVE.

    n_cell_dim: list(int)
        List containing the number of cells (for force computation) of cells in each
        direction.

    cell_side_length: list(float)
        List containing the side lengths of the cells for force computation in each
        direction.

    cell_list: list(`.Particle`)
        List containing the particles in each cell.

    matrix_phase: str
        Name of the matrix phase.

    file_paht: str
     File path that all outputs use.

    new_verlet_list: boolean
        Boolean signaling if there is a need to compute a new Verlet list

    temp_change_steps: list(int)
        List containing the time steps at wich the temperature was lowered

    max_residue: float
        Max residue allowed for the particles in the RVE
    """

    def __init__(self, particles, rve_dims):
        """
        Constructor for the RVE class.

        Parameters
        ----------
        particles: `-Particle`
            Particles inside the RVE.

        rve_dims: list(float)
            List containing the dimensions of the RVE.
        """
        self.particles = particles
        # Saving the particles in the RVE
        self.rve_dims = rve_dims
        # Saving the dimesnions of the RVE
        self.dim = len(rve_dims)
        # Dimension of the RVE
        self.box = Particle.box
        # Size of the simulation box
        self.volume_particles = Particle.volume
        # Initializing the total volume of all particles
        self.number_particles = Particle.number
        # Initializing the total number of particles
        self.volume_phase = Particle.volume_phase
        # Dictionary containing the volume occupied by each phase
        self.volume_RVE = Particle.volume_RVE
        # Volume of the RVE
        self.relative_energy_history = Particle.relative_energy_history
        # List containing the relative energy for each iteration
        self.kinetic_energy_history = Particle.kinetic_energy_history
        # List containing the kinetic energy for each iteration
        self.list_phases = Particle.list_phases
        # List containing the phases in the RVE
        self.n_cell_dim = Particle.n_cell_dim
        # List containing the number of cells in each direction
        self.matrix_phase = Particle.matrix_phase
        # Matrix phase of the RVE
        self.file_path = Particle.file_path
        # File path that all outputs use
        self.cell_side_length = Particle.cell_side_length
        # List containing the side lengths of the cells for force computation in each
        # direction
        self.cell_list = Particle.cell_list
        # List containing the particles in each cell
        self.new_verlet_list = Particle.new_verlet_list
        # Boolean signaling if there is a need to compute a new Verlet list
        self.temp_change_steps = Particle.temp_change_steps
        # List containing the time steps at wich the temperature was lowered
        self.max_residue = Particle.max_residue
        # Max residue allowed for the particles in the RVE
        self.total_overlap_history = Particle.total_overlap_history
        # History of the particles' overlap

    def useThisRVE(self, dp_dir):
        """Intialize the the Particle class attributes using this RVE."""
        Particle.box = self.box
        # Size of the simulation box
        Particle.volume = self.volume_particles
        # Initializing the total volume of all particles
        Particle.number = self.number_particles
        # Initializing the total number of particles
        Particle.volume_phase = self.volume_phase
        # Dictionary containing the volume occupied by each phase
        Particle.volume_RVE = self.volume_RVE
        # Volume of the RVE
        Particle.relative_energy_history = self.relative_energy_history
        # List containing the relative energy for each iteration
        Particle.kinetic_energy_history = self.kinetic_energy_history
        # List containing the kinetic energy for each iteration
        Particle.list_phases = self.list_phases
        # List containing the phases in the RVE
        Particle.n_cell_dim = self.n_cell_dim
        # List containing the number of cells in each direction
        Particle.matrix_phase = self.matrix_phase
        # Matrix phase of the RVE
        Particle.file_path = os.path.join(dp_dir, os.path.split(self.file_path)[0])
        # File path that all outputs use
        Particle.cell_side_length = self.cell_side_length
        # List containing the side lengths of the cells for force computation in each
        # direction
        Particle.cell_list = self.cell_list
        # List containing the particles in each cell
        Particle.new_verlet_list = self.new_verlet_list
        # Boolean signaling if there is a need to compute a new Verlet list
        Particle.temp_change_steps = self.temp_change_steps
        # List containing the time steps at wich the temperature was lowered
        Particle.max_residue = self.max_residue
        # Max residue allowed for the particles in the RVE
        Particle.total_overlap_history = self.total_overlap_history
        # History of the particles' overlap
        particles = self.particles
        # Particles in the RVE
        rve_dims = self.rve_dims
        # RVE dims
        return [particles, rve_dims]


class Particle():
    """
    This is the class for particles. Each particle in the RVE is an instance of this class. dsfsdfsdf
    During the simulation its class attributes characterize the current RVE.

    Attributes
    ----------
    center: list
        The position vector of the center of mass of the particle

    dim: int
        Number of the dimensions of the space where the particle "lives"

    force: array of floats
        Array containing the force acting on the particle.

    verlet_list: list(int)
        List containing the Verlet neighboors of the particle.

    phase: str
        Phase to which the particle belongs.

    Class Atributes
    ---------------
    box: list(float)
        Dimensions of the simulation box. Note: not necessarily equal to the RVE (e.g.
        cylindrical fibers).

    volume: float
        Total volume of the particles in the RVE.

    number: integer
        Number of particles in the RVE.

    volume_phase: list(float)
        List containing the volume of the particles in each phase.

    volume_RVE: float
        Volume of the RVE.

    relative_energy_history: list(float)
        List containing the relative energy at each iteration.

    kinetic_energy_history: list(float)
        List containing the kinetic energy at each iteration.

    list_phase: list(str)
        List containing the phases in the RVE.

    n_cell_dim: list(int)
        List containing the number of cells (for force computation) of cells in each
        direction.

    cell_side_length: list(float)
        List containing the side lengths of the cells for force computation in each
        direction.

    cell_list: list(`.Particle`)
        List containing the particles in each cell.

    matrix_phase: str
        Name of the matrix phase.

    file_paht: str
     File path that all outputs use.

    new_verlet_list: boolean
        Boolean signaling if there is a need to compute a new Verlet list

    temp_change_steps: list(int)
        List containing the time steps at wich the temperature was lowered

    max_residue: float
        Max residue allowed for the particles in the RVE

    total_overlap: float
        Overlap area/volme between the particles.

    total_overlap_history: list(float)
        History of the particle overlap
    """
    box = []
    # Size of the simulation box
    volume = 0
    # Initializing the total volume of all particles
    number = 0
    # Initializing the total number of particles
    volume_phase = {}
    # Dictionary containing the volume occupied by each phase
    volume_RVE = 0
    # Volume of the RVE
    relative_energy_history = []
    # List containing the relative energy for each iteration
    kinetic_energy_history = []
    # List containing the kinetic energy for each iteration
    list_phases = []
    # List containing the phases in the RVE
    n_cell_dim = []
    # List containing the number of cells in each direction
    cell_side_length = []
    # List containing the side lengths of the cells for force computation in each direction
    cell_list = []
    # List containing the particles in each cell
    matrix_phase = ''
    # Matrix phase of the RVE
    file_path = ''
    # File path that all outputs use
    new_verlet_list = False
    # Boolean signaling if there is a need to compute a new Verlet list
    temp_change_steps = []
    # List containing the time steps at wich the temperature was lowered
    max_residue = 0
    # Max residue allowed for the particles in the RVE
    total_overlap = 0
    # Overlap area/volme between the particles
    total_overlap_history = []
    # History of the particle overlap

    def __init__(self, dim, phase):
        '''
        The constructor for the Particle class.

        Parameters
        ----------
        dim: int
            Number of the dimensions of the space where the particle "lives".

        phase: string
            Phase to which the particle belongs.
        '''
        self.dim = dim
        # Setting the the dimension where the particle "lives"
        self.phase = phase
        # Phase to which the particle belongs
        self.force = np.zeros((dim))
        # Setting the initial force on the particle as zero
        self.verlet_list = []
        # Initializing the particles Verlet list
        Particle.volume += self.volume()
        # Updating the total volume
        try:
            if Particle.volume/Particle.volume_RVE > 1:
            # Checking if the volume fraction is below 1
                raise errors.VolumeFractionLargerOne(phase)
        except errors.VolumeFractionLargerOne as error:
            error.message()
            quit()
        Particle.number += 1
        # Updating the number of particles in the RVE
        Particle.volume_phase[self.phase] += self.volume()
        # Updating the volume corresponding to the particle's phase
        if Particle.matrix_phase != '':
        # The matrix phase has already been identified
            Particle.volume_phase[Particle.matrix_phase] = (
                1 - np.sum(list(Particle.volume_phase.values()))
                + Particle.volume_phase[Particle.matrix_phase])
            # Updating the volume occupied by the matrix

    def resetRVE():
        """Reset the class attributes for a new RVE simulation."""
        Particle.volume = 0
        # Initializing the total volume of all particles
        Particle.number = 0
        # Initializing the total number of particles
        Particle.volume_phase = {}
        # Dictionary containing the volume occupied by each phase
        Particle.relative_energy_history = []
        # List containing the relative energy for each iteration
        Particle.kinetic_energy_history = []
        # List containing the kinetic energy for each iteration
        Particle.n_cell_dim = []
        # List containing the number of cells in each direction
        Particle.cell_side_length = []
        # List containing the side lengths of the cells for force computation in each direction
        Particle.cell_list = []
        # List containing the particles in each cell
        Particle.file_path = ''
        # File path that all outputs use
        Particle.file_path = ''
        # File path that all outputs use
        Particle.new_verlet_list = False
        # Boolean signaling if there is a need to compute a new Verlet list
        Particle.temp_change_steps = []
        # List containing the time steps at wich the temperature was lowered


    def setPositionCenter(self, position):
        '''
        This function sets the position of the center of mass of the particle
        '''

        self.position_center = position
        # Setting the position of the center of mass of the particle

    def setVelocityCenter(self, velocity):
        '''
        This function sets the velocity of the center of mass of the particle.
        '''

        self.velocity_center = velocity
        # Setting the velocity of the center of mass of the particle

    def cleanForces(self):
        '''
        This function sets the forces acting on the particle to 0.
        '''

        self.force = np.zeros((self.dim))

    def cleanOverlapArea(self):
        '''
        This function sets the overlap area of the particle to 0.
        '''

        self.overlap_area = 0

    def intersectionVector(self, other_particle):
        """Compute the unit vector from the center of masss of particle i to particle j."""
        box = Particle.box
        # Saving the RVE dimensions
        vector_centers = other_particle.position_center - self.position_center
        vector_centers = vector_centers - box*np.round(vector_centers/box)
        # Vector connecting the centers of the current particle and the nearest image of
        # the other particle
        if self.dim == 2:
            angle_opposite = np.arctan2(vector_centers[1], vector_centers[0])
            if np.random.uniform() > 1:
                angle_new = angle_opposite + np.random.uniform(low=-np.pi/4, high=np.pi/4)
            else:
                angle_new = angle_opposite
            if np.linalg.norm(vector_centers) != 0:
                unit_vector_i_j = np.array([np.cos(angle_new), np.sin(angle_new)])
                # unit_vector_i_j = vector_centers/np.linalg.norm(vector_centers)
            else:
                random_vector = np.random.uniform(size=self.dim)
                unit_vector_i_j = random_vector/np.linalg.norm(random_vector)
            return unit_vector_i_j
        elif self.dim == 3:
            if np.linalg.norm(vector_centers) != 0:
                unit_vector_i_j = vector_centers/np.linalg.norm(vector_centers)
                # unit_vector_i_j = vector_centers/np.linalg.norm(vector_centers)
            else:
                random_vector = np.random.uniform(size=self.dim)
                unit_vector_i_j = random_vector/np.linalg.norm(random_vector)
            return unit_vector_i_j

# ==========================================================================================


class Ellipse(Particle):
    """
    This is the class for Ellipse.


    Attributes
    ----------
    phase: string
        Phase to which the ellipse belongs

    major_axis: float
        Major axis of the ellipse.

    semi_major_axis: float
        Half the major axis.

    minor_axis: float
        Minor axis of the ellipse.

    semi_minor_axis: float
        Half the minor axis.

    eccentricity: float
        Eccentricity of the ellipse.

    radius: float
        Radius of the circunscribed circle

    rot_mat: 2-array(floats)
        Rotation matrix from the global to the local coordinates.

    angle: float
        Angle in radians that the major axis forms with the x-axis.
    """


    def __init__(self, phase, major_axis, minor_axis, angle):
        '''
        This is the generator for the classe Ellipse.

        Parameters
        ----------
        phase: string
            Phase to which the ellipse belongs

        major_axis: float
            Major axis of the ellipse.

        minor_axis: float
            Minor axis of the ellipse.

        angle: float
            Angle in radians that the major axis forms with the x-axis
        '''
        self.major_axis = major_axis
        self.semi_major_axis = major_axis/2
        self.minor_axis = minor_axis
        self.semi_minor_axis = minor_axis/2
        self.angle = angle
        self.eccentricity = np.sqrt(1-minor_axis**2/major_axis**2)
        self.radius = major_axis/2
        self.rot_mat = np.array([[ np.cos(self.angle), np.sin(self.angle)],
                                 [-np.sin(self.angle), np.cos(self.angle)]])
        super().__init__(2, phase)

    def volume(self):
        '''
        This function computes the area(volume) of the ellipse.
        '''

        volume = np.pi*self.semi_major_axis*self.semi_minor_axis

        return volume

    def contract(self, distance):
        """Contract the particle."""
        self.semi_major_axis -= distance
        self.semi_minor_axis -= distance
        # Contracting the particle size subracting the minimum distance from the semi-axis
        self.major_axis = 2*self.semi_major_axis
        self.minor_axis = 2*self.semi_minor_axis
        self.eccentricity = np.sqrt(1 - self.minor_axis**2/self.major_axis**2)
        self.radius = self.semi_major_axis
        # Updating the other geometrical parameters

    def dilate(self, distance):
        """Dilate the particle."""
        self.semi_major_axis += distance
        self.semi_minor_axis += distance
        # Dilating the particle size adding the minimum distance to the semi-axis
        self.major_axis = 2*self.semi_major_axis
        self.minor_axis = 2*self.semi_minor_axis
        self.eccentricity = np.sqrt(1 - self.minor_axis**2/self.major_axis**2)
        self.radius = self.semi_major_axis
        # Updating the other geometrical parameters

    def pointInside(self, point, tol=1e-4, position='inside', verlet=False):
        '''
        Check if the point is inside, outside or on the ellipse given a tolerance.

        Parameters
        ---------
        self: `.Ellipse`
            Ellipse under analysis

        point: array
            Point under analysis

        tol: float
            Tolerance

        position: string
            'inside' or 'on'

        verlet: boolean
            Inside the ellipse itself or its neighboor, related to the Verlet list

        Returns
        -------
        point_in: bool
            True if the point is inside the ellipse and False otherwise.
        '''
        rot_mat = np.array([[ np.cos(self.angle), np.sin(self.angle)],
                            [-np.sin(self.angle), np.cos(self.angle)]])
        rot_mat_back = rot_mat.T
        # Rotation matrix that alignes ellipse 1 with the xy-axis
        r_vector = rot_mat.dot(point - self.position_center)
        # Defininig the radius vector relative to the coordinate system of the ellipse
        r_point = np.linalg.norm(r_vector)
        # Distance from the point to the center of the ellipse
        angle_pt_major = np.arctan2(r_vector[1], r_vector[0])
        # Angle that the vector connecting the center of the ellipse and the point makes
        # with the major axis
        if verlet:
        # Multiply the semi_minor_axis by the Verlet factor
            semi_minor_axis = self.semi_minor_axis*(1 - Particle.verlet_factor)
            # Semi minor axis of the Verlet neighboorhood
        else:
            semi_minor_axis = self.semi_minor_axis
            # Semi minor axis of the original ellipse
        if position=='inside':
        # Checking if the point is inside the ellipse
            point_in = \
                r_point <= tol + \
                    semi_minor_axis/np.sqrt(1-(self.eccentricity*np.cos(angle_pt_major))**2)
            # Using the polar form of the ellipse checking if the point is inside the ellipse
        elif position=='on':
        # Checking if the point is on the ellipse
            point_in = \
                np.abs(r_point - \
                    semi_minor_axis/np.sqrt(1-(self.eccentricity*np.cos(angle_pt_major))**2))\
                    < tol
            # Using the polar form of the ellipse checking if the point is inside the ellipse
        return point_in

    def intersectionAreaEllipseEllipse(self, other_ellipse):
        """Compute the orverlap area between the current and the other ellipse."""
        box = Particle.box
        # Saving the RVE dimensions
        diff_in_box = self.position_center - other_ellipse.position_center
        # Difference vector between the center of the two ellipses
        diff_nearest_other = box*np.round(diff_in_box/box)
        # Vector from the position of the other ellipse to its nearest image to the current
        # ellipse
        intersect_pts = intersectionPointsEllipses(
            self.major_axis/2, self.minor_axis/2,  self.position_center, self.angle,
            other_ellipse.major_axis/2, other_ellipse.minor_axis/2,
            other_ellipse.position_center + diff_nearest_other, other_ellipse.angle)
        # Computing the intersection points of the two ellipses
        if len(intersect_pts) == 0:
        # Either the ellipses are disjoint or one of them is completly inside the other
            if self.volume() >= other_ellipse.volume():
            # The current ellipse is larger than the other ellipse
                if self.pointInside(other_ellipse.position_center):
                # The other ellipse is completly inside the current ellipse
                    intersection_area = other_ellipse.volume()
                    # The intersection area is the area of the smaller ellipse
                else:
                # The ellipses are disjoint
                    intersection_area = 0
                    # The intersection area is 0
            else:
                if other_ellipse.pointInside(self.position_center):
                # The current ellipse is completly inside the other ellipse
                    intersection_area = self.volume()
                    # The intersection area is the area of the smaller ellipse
                else:
                # The ellipses are disjoint
                    intersection_area = 0
                    # The intersection area is 0
        elif len(intersect_pts) == 1:
        # Either the ellipses are disjoint or one of them is completly inside the other,
        # except for the intersection point
            if self.volume() >= other_ellipse.volume():
            # The current ellipse is larger than the other ellipse
                if self.pointInside(other_ellipse.position_center):
                # The other ellipse is completly inside the current ellipse
                    intersection_area = other_ellipse.volume()
                    # The intersection area is the area of the smaller ellipse
                else:
                # The ellipses are disjoint
                    intersection_area = 0
                    # The intersection area is 0
            else:
                if other_ellipse.pointInside(self.position_center):
                # The current ellipse is completly inside the other ellipse
                    intersection_area = self.volume()
                    # The intersection area is the area of the smaller ellipse
                else:
                # The ellipses are disjoint
                    intersection_area = 0
                    # The intersection area is 0
        elif len(intersect_pts)==2:
        # The ellipses intersect in two points. The case where one of the ellipses is
        # inside the other and both are tangent at the intersection points is disregarded
            intersection_area = 0
            # Initializing the intersection area
            intersect_pts_ord = self.sortPointsOnEllipse(intersect_pts)
            # Ordering the intersection points according to their angle relative to the
            # major axis of the current ellipse counter clockwise
            ellipses = [self, other_ellipse]
            # Saving the ellipses in a list
            midpoint = self.midpointOnEllipse(intersect_pts_ord[0], intersect_pts_ord[1])
            # Midpoint between the first two intersection points in the current ellipse
            if other_ellipse.pointInside(midpoint - diff_nearest_other):
            # If the midpoint is on the other ellipse
                intersection_area += \
                    self.areaEllipseSection(intersect_pts_ord[0], intersect_pts_ord[1])
                # The correct segment belongs to the current ellipse
                k_ellipse = 0
                # Index of the current ellipse
            else:
                intersection_area += \
                    other_ellipse.areaEllipseSection(
                        intersect_pts_ord[0] - diff_nearest_other,
                        intersect_pts_ord[1] - diff_nearest_other)
                # The correct segment belongs to the other ellipse
                k_ellipse = 1
                # Index of the other ellipse
            for i_segment in range(1, 2):
            # Running through each segment
                k_ellipse = np.mod(k_ellipse + 1, 2)
                # Index of the ellipse whose area segment needs to calculated
                intersection_area += \
                    ellipses[k_ellipse].areaEllipseSection(
                        intersect_pts_ord[np.mod(i_segment,2)] - k_ellipse*diff_nearest_other, \
                        intersect_pts_ord[np.mod(i_segment+1,2)] - k_ellipse*diff_nearest_other)
                # Computing the area of the segment
        elif len(intersect_pts)==3:
        # This case is disregarded
            intersection_area = 0
        elif len(intersect_pts)==4:
        # One of the ellipses goes through the other
            intersection_area = 0
            # Initializing the intersection area
            intersect_pts_ord = self.sortPointsOnEllipse(intersect_pts)
            # Ordering the intersection points according to their angle relative to the
            # major axis of the current ellipse counter clockwise
            intersection_area += 0.5*np.abs(
                (intersect_pts_ord[2][0]-intersect_pts_ord[0][0])
                * (intersect_pts_ord[3][1]-intersect_pts_ord[1][1])
                - (intersect_pts_ord[3][0]-intersect_pts_ord[1][0])
                * (intersect_pts_ord[2][1]-intersect_pts_ord[0][1]))
            # Computing the area of the quadrilateral inscribed in the overlap of the
            # two ellipses
            ellipses = [self, other_ellipse]
            # List of the ellipse objects to iterate over
            midpoint = self.midpointOnEllipse(intersect_pts_ord[0], intersect_pts_ord[1])
            # Obtaining the midpoint between the first two intersection points to decide
            # to which ellipses belong to the area sections to be calculated
            if other_ellipse.pointInside(midpoint - diff_nearest_other):
                intersection_area += \
                    self.areaEllipseSection(intersect_pts_ord[0], intersect_pts_ord[1])
                k_ellipse = 0
            else:
                intersection_area += \
                    other_ellipse.areaEllipseSection(
                        intersect_pts_ord[0] - diff_nearest_other,
                        intersect_pts_ord[1] - diff_nearest_other)
                k_ellipse = 1
            for i_segment in range(1,4):
            # Running through each segment
                k_ellipse = np.mod(k_ellipse + 1, 2)
                intersection_area += \
                    ellipses[k_ellipse].areaEllipseSection(
                        intersect_pts_ord[np.mod(i_segment, 4)]
                                          - k_ellipse*diff_nearest_other, \
                        intersect_pts_ord[np.mod(i_segment + 1, 4)]
                                          - k_ellipse*diff_nearest_other)
        return intersection_area

    def midpointOnEllipse(self, *args):
        '''
        This function returns the point midway between point_1 and point_2, anti clockwise.
        '''

        rot_mat = np.array([[ np.cos(self.angle), np.sin(self.angle)],
                            [-np.sin(self.angle), np.cos(self.angle)]])
        rot_mat_back = rot_mat.T
        # Rotation matrix that alligns ellipse 1 with the xy-axis
        angle = []
        # Initializing the list containing the angles of the points relative to the
        # center of the ellipse with axis coinciding with the major and minor axis of the
        # ellipse
        for i_point in args:
        # Running through all the points
            radius_vector = rot_mat.dot(i_point-self.position_center)
            # Obtaining the radius vector corresponding to the i_point in the coordinate
            # system of the ellipse
            angle_i = np.arctan2(radius_vector[1],radius_vector[0])
            # Angle the radius vector of the point makes with the major axis of the ellipse
            # between 0 and pi
            if angle_i<0:
            # If the y-coordinate of the radius vector is negative
                angle_i = angle_i + 2*np.pi
                # Accounting for the fact that arccos only gives values between 0 and pi
            angle.append(angle_i)
        angle_mid = (angle[0] + angle[1])/2
        # Angle of the midpoint
        radius_mid = self.semi_minor_axis/np.sqrt(1-(self.eccentricity*np.cos(angle_mid))**2)
        # Radius of the midpoint
        midpoint_loc = radius_mid*np.array([np.cos(angle_mid), np.sin(angle_mid)])
        # Cartesian coordinates of the midpoint in the coordinate system of the ellipse
        midpoint = self.position_center + rot_mat.T.dot(midpoint_loc)
        # Cartesian coordinates of the midpoint in the global coordinate system
        return midpoint

    def sortPointsOnEllipse(self, points):
        '''
        This function sorts the points given in the ellipse clockwise.
        '''

        rot_mat = np.array([[ np.cos(self.angle), np.sin(self.angle)],
                            [-np.sin(self.angle), np.cos(self.angle)]])
        rot_mat_back = rot_mat.T
        # Rotation matrix that alligns ellipse 1 with the xy-axis
        angle = []
        # Initializing the list containing the angles of the points relative to the
        # center of the ellipse with axis coinciding with the major and minor axis of the
        # ellipse
        for i_point in points:
        # Running through all the points
            radius_vector = rot_mat.dot(i_point-self.position_center)
            # Obtaining the radius vector corresponding to the i_point in the coordinate
            # system of the ellipse
            angle_i = np.arctan2(radius_vector[1],radius_vector[0])
            # Angle the radius vector of the point makes with the major axis of the ellipse
            # between 0 and pi
            if angle_i<0:
            # If the y-coordinate of the radius vector is negative
                angle_i = angle_i + 2*np.pi
                # Accounting for the fact that arccos only gives values between 0 and pi
            angle.append(angle_i)
            # Appending the angle
        y_ordered = [points[i] for i in np.argsort(angle)]
        # Obtaining the list of points with angles sorted counter clockwise
        return y_ordered

    def areaEllipseSection(self, intersect_pt_1, intersect_pt_2):
        '''
        This function computes the area of the segment defined by the secant line drawn
        between the two points given and the ellipse, anti clockwise from point 1 to point 2.

        Parameters:
            self: Ellipse
                Ellipse under analysis.
            intersect_pt_1: array
                Array containing the coordinates of the first intersection point. The
                funtion does not check if the point is indeed on the ellipse
            intersect_pt_2: array
                Array containing the coordinates of the second intersection point. The
                funtion does not check if the point is indeed on the ellipse

        Returns:
            area_segment: float
                Area of the segment defined by the secant line drawn between the two
                points given and the ellipse
        '''


        rot_mat = np.array([[ np.cos(self.angle), np.sin(self.angle)],
                            [-np.sin(self.angle), np.cos(self.angle)]])
        # Rotation matrix
        pt_1 = rot_mat.dot(intersect_pt_1 - self.position_center)
        pt_2 = rot_mat.dot(intersect_pt_2 - self.position_center)
        # Translation and rotation of the ellipse to the origin aligning with the xy axis
        if pt_1[1]>0:
            theta_1 =  np.arccos(np.max([np.min([pt_1[0]/self.semi_major_axis, 1]), -1]))
        else:
            theta_1 = 2*np.pi - np.arccos(np.max([np.min([pt_1[0]/self.semi_major_axis, 1]), -1]))
        # Computing the parametric angle corresponding to the first intersection point
        # ensuring that there are no errors using the trigonometric functions
        if pt_2[1]>0:
            theta_2 = np.arccos(np.max([np.min([pt_2[0]/self.semi_major_axis,1]),-1]))
        else:
            theta_2 = 2*np.pi - np.arccos(np.max([np.min([pt_2[0]/self.semi_major_axis,1]),-1]))
        # Computing the parametric angle corresponding to the second intersection point
        if theta_1<=theta_2:
            theta_1_hat = theta_1
        else:
            theta_1_hat = theta_1 - 2*np.pi
        # Ensuring that the angle theta_1 is always smaller than theta_2 as the area
        # is computed in an anti-clockwise manner from point 1 to 2
        area_sector = (theta_2 - theta_1_hat)*self.semi_major_axis*self.semi_minor_axis/2
        # Area of the ellipse sector defined by the two points
        area_triangle_sgn = np.sign(theta_2 - theta_1_hat - np.pi)/2*np.abs(
            pt_1[0]*pt_2[1]-pt_2[0]*pt_1[1])
        # Signed area of the triangle defined by the two point and the center of the
        # ellipse
        area_segment = area_sector + area_triangle_sgn
        # Area of the ellipse segment
        return area_segment

    def intersectionArea(self, other_particle):
        '''
        This function computes the intersection between the ellipse and the other particle.

        Parameters:
            other_particle: Particle
                Other particle
        '''
        intersection_area = self.intersectionAreaEllipseEllipse(other_particle)
        # Computing the intersection area
        return intersection_area
        # Returning the intersection area

    def intersectionVerlet(self, other_particle):
        '''
        This function computes the intersection between the disk and the other particle.

        Parameters:
            other_particle: Particle
                Other particle
        '''
        intersection_verlet = self.intersectionVerletEllipseEllipse(other_particle)
        # Computing the intersection area
        return intersection_verlet
        # Returning the intersection area

    def intersectionVerletEllipseEllipse(self, other_ellipse):
        '''
        This function computes the intersection area between two disks
        '''

        box = Particle.box

        diff_in_box = self.position_center - other_ellipse.position_center
        # Difference vector between the center of the two ellipses
        diff_nearest_other = box*np.round(diff_in_box/box)
        # Difference vector to the nearest image of the other particle
        y_inter_sect = intersectionPointsEllipses(
            Particle.verlet_factor*self.semi_major_axis, Particle.verlet_factor*self.semi_minor_axis, self.position_center,
            self.angle, Particle.verlet_factor*other_ellipse.semi_major_axis, Particle.verlet_factor*other_ellipse.semi_minor_axis,
            other_ellipse.position_center+ diff_nearest_other, other_ellipse.angle)
        if len(y_inter_sect)>0:
        # There are intersection points betweeen the two neighboorhoods
            intersection_verlet = True
        else:
        # Either the ellipses are disjoint or one of them is completly inside the other
            if self.volume() >= other_ellipse.volume():
            # The current ellipse is larger than the other ellipse
                if self.pointInside(other_ellipse.position_center):
                # The other ellipse is completly inside the current ellipse
                    intersection_verlet = True
                    # The intersection area is the area of the smaller ellipse
                else:
                # The ellipses are disjoint
                    intersection_verlet = False
                    # The intersection area is 0
            else:
                if other_ellipse.pointInside(self.position_center):
                # The current ellipse is completly inside the other ellipse
                    intersection_verlet = True
                    # The intersection area is the area of the smaller ellipse
                else:
                # The ellipses are disjoint
                    intersection_verlet = False
                    # The intersection area is 0
        return intersection_verlet

    def insideVerlet(self):
        """Check if the ellipse has moved outside its Verlet neighboorhood."""
        if np.linalg.norm(self.displacement_last_verlet) >= \
            self.semi_minor_axis*(Particle.verlet_factor - 1):
        # Its possible for the ellipse to have moved outside its Verlet neighboorhood
            point_in = self.pointInside(
                self.displacement_last_verlet + self.position_center, verlet=True)
            # Checking if the ellipse is still inside its Verlet neighboorhood
        else:
        # the center of the ellipse has not
            point_in = True

        return point_in

    def generatePointsOnSurface(self, n_points, erosion_thick=0):
        """Generate *n_points* on the surface of the ellipse."""
        points_loc = np.array([[self.semi_major_axis*np.cos(theta), self.semi_minor_axis*np.sin(theta)]
                              for theta in np.linspace(0, 2*np.pi, n_points, endpoint=False)])
        # Generating the points in the Disk's local coordinates
        if erosion_thick > 0:
        # If erosion was sepcified
            for point_ind, _ in enumerate(points_loc):
            # For each point on the surface with its corresponding homogeneous angle
                angle_normal = np.arctan2(self.semi_major_axis/self.semi_minor_axis*points_loc[point_ind][1], points_loc[point_ind][0])
                # Computing the angle of the normal at the current point
                points_loc[point_ind] -= erosion_thick*np.array([np.cos(angle_normal), np.sin(angle_normal)])
                # Translation of the point in the normal direction to the surface by the
                # specified thickness (erosion)
        points_glob = np.array([self.rot_mat.T.dot(point_loc) + self.position_center for point_loc in points_loc])
        # Transforming local in global coordinates
        return points_glob

    def computeCriticalErosionThickness(self):
        """Compute the critical erosion thickness for an ellipse."""
        erosion_thickness = self.semi_minor_axis**2/self.semi_major_axis
        # Semi-latus rectum
        return erosion_thickness


class Disk(Ellipse):
    '''
    This is the subclass of particles with the form of a circular disk.

    Attributes
    ----------
    radius: float
        Radius of the disk
    '''

    def __init__(self, phase, radius):
        '''
        The constructor of the Disk particle.

        Parameters
        ----------
        center: array
            The position vector of the center of mass of the particle

        dim: int
            Number of the dimensions of the space where the particle "lives"

        radius: float
            Radius of the disk
        '''
        self.dim = 2
        self.radius = radius
        self.major_axis = 2*radius
        self.minor_axis = 2*radius
        self.semi_major_axis = radius
        self.semi_minor_axis = radius
        self.angle = 0
        self.force = np.zeros((self.dim), dtype='float')
        self.n_cell_dim = []
        self.verlet_list = []
        Particle.volume += self.volume()
        Particle.number += 1
        self.phase = phase

    def generatePointsOnSurface(self, n_points, erosion_thick=0):
        """Generate *n_points* on the surface of the Disk."""
        points_loc = np.array([[self.radius*np.cos(theta), self.radius*np.sin(theta)]
                              for theta in np.linspace(0, 2*np.pi, n_points, endpoint=False)])
        # Generating the points in the Disk's local coordinates
        points_glob = points_loc + self.position_center
        # Transforming local in global coordinates
        if erosion_thick > 0:
        # If erosion was sepcified
            for (point_ind, _), theta in zip(enumerate(points_glob), np.linspace(0, 2*np.pi, n_points, endpoint=False)):
            # For each point on the surface with its corresponding homogeneous angle
                points_glob[point_ind] -= erosion_thick*np.array([np.cos(theta), np.sin(theta)])
                # Translation of the point in the normal direction to the surface by the
                # specified thickness (erosion)
        return points_glob

    def intersectionArea(self, other_particle):
        '''
        This function computes the intersection between the disk and the other particle.

        Parameters:
            other_particle: Particle
                Other particle
        '''
        class_name_other_particle = other_particle.__class__.__name__
        # Saving the class name of the other particle as a string
        if 'Disk' == class_name_other_particle or 'CylindricalFiber' == class_name_other_particle:
            # The other particle is also a Disk
            intersection_area = self.intersectionAreaDiskDisk(other_particle)
            # Computing the intersection area
            return intersection_area
            # Returning the intersection area
        elif 'Ellipse' == class_name_other_particle:
        # The other particle is an Ellipse
            intersection_area = other_particle.intersectionAreaEllipseEllipse(self)
            # Computing the intersection area
            return intersection_area
            # Returning the intersection area

    def intersectionAreaDiskDisk(self, other_disk):
        '''
        This function computes the intersection area between two disks
        '''
        box = Particle.box
        # Saving the simulation box
        diff_center = self.position_center - other_disk.position_center
        diff_center = diff_center - box*np.round(diff_center/box)
        # Vector from the current disk to the nearest image of the other disk
        d = np.sqrt(diff_center.dot(diff_center))
        # Distance between the center of the disks
        if self.radius >= other_disk.radius:
        # The radius of the self is larger than the radius of the other disk
            r_1 = self.radius
            # Disk 1 is the disk with the larger radius
            r_2 = other_disk.radius
            # Disk 2 is the disk with the smaller radius
        else:
        # The radius of the other disk is larger than the radius of the self
            r_1 = other_disk.radius
            # Disk 1 is the disk with the larger radius
            r_2 = self.radius
            # Disk 2 is the disk with the smaller radius
        if d >= (r_1 + r_2):
        # The disks intersect at most at one point
            intersection_area = 0
            # The intersection area of the disks is zero
        elif d <= r_1 - r_2:
        # Disk 2 is interely contained within Disk 1
            intersection_area = np.pi*r_2**2
            # The intersection area is equal to the area of the smaller disk, Disk 2
        else:
            d_1 = (r_1**2 - r_2**2 + d**2)/(2*d)
            # x coordinate of the intersection point of the two disks if the the origin is at
            # disk 1 and the x axis goes through the center of both disks
            d_2 = d - d_1
            # Distance in the x axis from the intersection point to disk 2
            intersection_area = r_1**2*np.arccos(d_1/r_1) - d_1*np.sqrt(r_1**2-d_1**2) + \
                r_2**2*np.arccos(d_2/r_2) - d_2*np.sqrt(r_2**2 - d_2**2)
            # Computing the intersection area
        return intersection_area
        # Returning the intersection area


    def intersectionVerlet(self, other_particle):
        '''
        This function computes the intersection between the disk and the other particle.

        Parameters:
            other_particle: Particle
                Other particle
        '''
        class_name_other_particle = other_particle.__class__.__name__
        # Saving the class name of the other particle as a string
        if 'Disk' == class_name_other_particle or 'CylindricalFiber' == class_name_other_particle:
        # The other particle is also a Disk
            intersection_verlet = self.intersectionVerletDiskDisk(other_particle)
            # Computing the intersection area
            return intersection_verlet
            # Returning the intersection area
        elif 'Ellipse' == class_name_other_particle:
        # The other particle is an Ellipse
            intersection_verlet = other_particle.intersectionVerletEllipseEllipse(self)
            # Computing the intersection area
            return intersection_verlet
            # Returning the intersection area

    def pointInside(self, point):

        if np.linalg.norm(self.position_center - point) <= self.radius:
            point_in = True
        else:
            point_in = False

        return point_in

    def intersectionVerletDiskDisk(self, other_disk):
        '''
        This function computes the intersection area between two disks
        '''

        box = Particle.box
        # Saving the limits of the box
        diff_center = self.position_center - other_disk.position_center
        diff_center = diff_center - box*np.round(diff_center/box)
        # Vector between the centers of the current disk and the nearest image of the other
        # disk
        d = np.sqrt(diff_center.dot(diff_center))
        # Distance between the disks
        if d < (self.radius+other_disk.radius)*Particle.verlet_factor:
        # The disks are in eachothers neighboorhoods
            intersection_verlet = True
        else:
            intersection_verlet = False
        return intersection_verlet

    def volume(self):
        '''
        This function computes the volume/area of the disk.
        '''

        volume = np.pi*self.radius**2

        return volume

    def insideVerlet(self):
        """Check if the ellipse has moved outside its Verlet neighboorhood."""
        if np.linalg.norm(self.displacement_last_verlet) >= \
           self.radius*(Particle.verlet_factor - 1):
        # Its possible for the ellipse to have moved outside its Verlet neighboorhood
            point_in = False
            # Checking if the ellipse is still inside its Verlet neighboorhood
        else:
        # the center of the ellipse has not
            point_in = True
        return point_in

    def computeCriticalErosionThickness(self):
        """Compute the critical erosion thickness for a disk."""
        erosion_thickness = self.radius
        return erosion_thickness

class CylindricalFiber(Disk):
    '''
    This is the subclass of particles with the form of a circular disk.

    Attributes
    ----------
    radius: float
        Radius of the disk
    '''

    def __init__(self, phase, radius, direction, rve_dims):
        """
        The constructor of the cylindrical fiber particle.

        Parameters
        ----------
        phase: string
            Phase to which the particle belongs to.

        radius: float
            Radius of the disk
        """
        self.direction_fibers = direction
        # Integer giving the direction of the fibers
        self.length_dir_fibers = rve_dims[self.direction_fibers]
        # Length of the fibers
        rve_dims = np.delete(rve_dims, self.direction_fibers)
        Particle.box = rve_dims
        # Setting the size of the simulation box
        super().__init__(phase, radius)
        # Using the constructor of the parent class


    def volume(self):
        """Compute the volume of the cylindrical fiber."""

        volume = np.pi*self.radius**2*self.length_dir_fibers

        return volume


class Ellipsoid(Particle):
    """docstring for Ellipsoid."""

    def __init__(self, phase, axis_1, axis_2, axis_3, euler_angle_x, euler_angle_y,
                 euler_angle_z, angle):
        '''
        This is the generator for the classe Ellipse.

        Parameters
        ----------
        phase: string
            Phase to which the ellipsoid belongs

        axis_1: float
            Length of the ellipsoid axis along the local (pre-rotation) x1-axis.

        axis_2: float
            Length of the ellipsoid axis along the local (pre-rotation) x2-axis.

        axis_3: float
            Length of the ellipsoid axis along the local (pre-rotation) x3-axis.

        euler_angle_x: float
            Euler angle relative to the local x1 (pre-rotation) axis.

        euler_angle_y: float
            Euler angle relative to the local x2 (pre-rotation) axis.

        euler_angle_z: float
            Euler angle relative to the local x3 (pre-rotation) axis.

        angle: float
            Angle in radians that axis x1 and x2 rotate arround the x3 axis.
        '''
        self.axis_1 = axis_1
        self.semi_axis_1 = axis_1/2
        self.axis_2 = axis_2
        self.semi_axis_2 = axis_2/2
        self.axis_3 = axis_3
        self.semi_axis_3 = axis_3/2
        self.rotation_axis = (np.array([euler_angle_x, euler_angle_y, euler_angle_z])
                              / np.linalg.norm(np.array([euler_angle_x, euler_angle_y,
                                                         euler_angle_z])))

        self.angle = angle
        self.rot_quat = np.array([np.cos(angle/2),
                                  np.sin(angle/2)*self.rotation_axis[0],
                                  np.sin(angle/2)*self.rotation_axis[1],
                                  np.sin(angle/2)*self.rotation_axis[2]])
        self.radius = np.max([self.semi_axis_1, self.semi_axis_3, self.semi_axis_3])
        # Radius of the circunscribed sphere
        q = self.rot_quat
        self.rotation_mat = np.array([[1-2*(q[2]**2+q[3]**2), 2*(q[1]*q[2]-q[3]*q[0]),
                                       2*(q[1]*q[3]-q[2]*q[0])],
                                      [2*(q[1]*q[2]-q[3]*q[0]), 1-2*(q[1]**2+q[3]**2),
                                       2*(q[2]*q[3]-q[1]*q[0])],
                                      [2*(q[1]*q[3]-q[2]*q[0]), 2*(q[2]*q[3]-q[1]*q[0]),
                                       1-2*(q[1]**2+q[2]**2)]])
        # Rotation matrix from local to global coordinates
        super().__init__(3, phase)

    def contract(self, distance):
        """Contract the particle."""
        self.semi_axis_1 -= distance
        self.semi_axis_2 -= distance
        self.semi_axis_3 -= distance
        # Contracting the particle size subracting the minimum distance from the semi-axis
        self.axis_1 = 2*self.semi_axis_1
        self.axis_2 = 2*self.semi_axis_2
        self.axis_3 = 2*self.semi_axis_3
        self.radius = np.max([self.semi_axis_1, self.semi_axis_3, self.semi_axis_3])
        # Updating the other geometrical parameters

    def dilate(self, distance):
        """Dilate the particle."""
        self.semi_axis_1 += distance
        self.semi_axis_2 += distance
        self.semi_axis_3 += distance
        # Dilating the particle size adding the minimum distance to the semi-axis
        self.axis_1 = 2*self.semi_axis_1
        self.axis_2 = 2*self.semi_axis_2
        self.axis_3 = 2*self.semi_axis_3
        self.radius = np.max([self.semi_axis_1, self.semi_axis_3, self.semi_axis_3])
        # Updating the other geometrical parameters

    def M(self):
        M = np.concatenate((np.concatenate((self.rotation_mat, np.array([self.position_center]).T),
            axis=1), np.array([[0., 0., 0., 1.]])), axis=0)
        return M

    def M_inv(self, diff_nearest=np.array([0., 0., 0.])):
        M_inv = np.concatenate((np.concatenate((self.rotation_mat.T,
            np.array([-self.rotation_mat.T.dot(self.position_center + diff_nearest)]).T), axis=1),
            np.array([[0., 0., 0., 1.]])), axis=0)
        return M_inv

    def A_glob(self, verlet=False, diff_nearest=np.array([0., 0., 0.])):
        if verlet == True:
            [semi_axis_1, semi_axis_2, semi_axis_3] = (Particle.verlet_factor
                *np.array([self.semi_axis_1, self.semi_axis_2, self.semi_axis_3]))
        else:
            [semi_axis_1, semi_axis_2, semi_axis_3] = (
                [self.semi_axis_1, self.semi_axis_2, self.semi_axis_3])
        A_loc = np.array([[1./semi_axis_1**2, 0., 0., 0.],
                          [0., 1./semi_axis_2**2, 0., 0.],
                          [0., 0., 1./semi_axis_3**2, 0.],
                          [0., 0., 0., -1.]], dtype=float)
        A_glob = self.M_inv(diff_nearest).T.dot(A_loc.dot(self.M_inv(diff_nearest)))
        return A_glob

    def volume(self):
        '''
        This function computes the area(volume) of the ellipse.
        '''

        volume = 4/3*np.pi*self.semi_axis_1*self.semi_axis_2*self.semi_axis_3

        return volume

    def pointInside(self, point, tol=1e-6, position='inside', verlet=False):
        """
        Check if a given point is inside the ellipsoid.

        This function determines if the point is inside, outside or on the ellipse given a
        tolerance.

        Parameters
        ----------
        self: `.Ellipsoid`
            Ellipsoid under analysis
        point: array
            Point under analysis
        tol: float
            Tolerance
        position: string
            'inside' or 'on'
        verlet: boolean
            Inside the ellipse itself or its neighboor, related to the Verlet list

        Returns
        -------
        point_in: bool
            True if the point is inside the ellipse and False otherwise.
        """
        rot_mat_l_g = self.rotation_mat
        # Rotation matrix from local to global coordinates
        point_loc = rot_mat_l_g.T.dot(point - self.position_center)
        # Point in local coordinates
        if verlet:
        # Multiply the semi_minor_axis by the Verlet factor
            semi_axis_1 = self.semi_axis_1*(1 - Particle.verlet_factor)
            semi_axis_2 = self.semi_axis_2*(1 - Particle.verlet_factor)
            semi_axis_3 = self.semi_axis_3*(1 - Particle.verlet_factor)
            # Semi minor axis of the Verlet neighboorhood
        else:
            semi_axis_1 = self.semi_axis_1
            semi_axis_2 = self.semi_axis_2
            semi_axis_3 = self.semi_axis_3
            # Semi minor axis of the original ellipse
        if position == 'inside':
        # Checking if the point is inside the ellipse
            point_in = (point_loc[0]**2/semi_axis_1**2 + point_loc[1]**2/semi_axis_2**2
                        + point_loc[2]**2/semi_axis_3**2 - 1 <= tol)
            # Using the polar form of the ellipse checking if the point is inside the ellipse
        elif position == 'on':
        # Checking if the point is on the ellipse
            point_in = (np.abs(point_loc[0]**2/semi_axis_1**2
                               + point_loc[1]**2/semi_axis_2**2
                               + point_loc[2]**2/semi_axis_3**2 - 1) <= tol)
            # Using the polar form of the ellipse checking if the point is inside the ellipse
        return point_in


    def intersectionVolumeEllipsoidOther(self, other_particle, type='random', tol=5, max_it=1000,
                                         seq_size=50):
        """
        Compute the overlap volume between this ellipsoid and another particle.

        This function uses Monte Carlo method to obtain the overlap volume, generating
        random points inside the current ellipsoid and checking if they also belong to the
        other particle. The overlap volume is proportional to the probability of the point
        being inside both particles.

        Parameters
        ----------
        self: `.Ellipsoid`
            Current ellipsoid

        other_particle: `.Particle`
            Other particle.

        type: {'random', 'regular'}, optional
            Integration method. 
            'random' - Monte Carlo method
            'regular' - Quadrature (scipy)

        tol: float, optional
            Tolerance for the error estimate.

        max_it: integer, optional
            Maximum number of iterations.

        seq_size: integer, optional
            Number of points used to estimate the overlap volume at each iteration

        Returns
        -------
        overlap_volume: float
            Overlap volume of the two particles.
        """
        box = Particle.box
        # Saving the RVE dimensions
        diff_in_box = self.position_center - other_particle.position_center
        # Difference vector between the center of the two ellipses
        diff_nearest_other = box*np.round(diff_in_box/box)
        # Vector between the other particle in the RVE to its nearest image to the current
        # ellipsoid
        if type == 'random':
            k_iteration = 0
            # Initializing the iteration counter
            overlap_volume_est = []
            # Vector of the overlap volume estimates
            error = 10
            # Initializing the error
            while (error > tol) and (k_iteration < max_it):
            # Run the Monte Carlo method while the error estimate is larger than the tolerance
            # and the number of iterations is smaller than the maximm allowed number of
            # iterations
                total_n_points = 0
                points_inside = 0
                # Initializing the counters for the number of points generated and the number of
                # points inside both volumes
                for i_point in range(seq_size):
                # Generating seq_size points
                    total_n_points += 1
                    # Counting the generated points
                    point = self.generatePointInside()
                    # Generating a random point inside the current ellipsoid
                    if other_particle.pointInside(point - diff_nearest_other):
                    # If the generated point is inside the volume of the other particle
                        points_inside += 1
                        # Counting the points inside both particles
                overlap_volume_est.append(self.volume()*points_inside/total_n_points)
                # Estimation for the overlap volume
                k_iteration += 1
                # Increasing the iteration couter
                if k_iteration > 2:
                # If there are more than 2 estimations
                    overlap_volume = np.mean(overlap_volume_est)
                    error = (np.std(overlap_volume_est)/np.sqrt(len(overlap_volume_est))
                             /overlap_volume * 100)
                    # Estimation and error computed assuming that each iteration is independent
                    # from the last and follow a normal distribution
        elif type == 'regular':
            A = self.semi_axis_1
            B = self.semi_axis_2
            C = self.semi_axis_3

            def pointsInside(x, y, z):
                pointIn = other_particle.pointInside(self.rotation_mat.dot([x, y, z]) + self.position_center - diff_nearest_other)
                if pointIn:
                    value = 1
                else:
                    value = 0
                return value

            (overlap_volume, _) = integrate.tplquad(pointsInside, -A, A,
                lambda x: -B*np.sqrt(1 - x**2/A**2),
                lambda x: B*np.sqrt(1 - x**2/A**2),
                lambda x, y: -C*np.sqrt(1 - x**2/A**2 - y**2/B**2),
                lambda x, y: C*np.sqrt(1 - x**2/A**2 - y**2/B**2), epsrel=0.1)

        return overlap_volume

    def generateRegularGrid(self, n_samples):
        """Generate a regular sample of points in the ellipsoid."""
        n_theta = int(np.sqrt(n_samples**(1)))
        n_phi = int(np.cbrt(n_samples**(1)))
        # Number of sample points for the angle
        n_r = int(np.round(n_samples/n_theta/n_phi))
        # Number of sample points for the radius. Muliplied by the number of points for the
        # angle gives the number of sample points
        radius = (np.linspace(0.01, 1, n_r, endpoint=True))**(1/3)
        theta = np.linspace(0, np.pi, n_theta, endpoint=False)
        phi = np.linspace(0, 2*np.pi, n_phi, endpoint=False)
        # Regularly and uniformly sampling the angle and the radius
        x_samples = []
        for i_theta in theta:
            for j_phi in phi:
                for k_radius in radius:
                    x_loc = np.array(
                        [k_radius*self.semi_axis_1*np.sin(i_theta)*np.cos(j_phi),
                         k_radius*self.semi_axis_2*np.sin(i_theta)*np.sin(j_phi),
                         k_radius*self.semi_axis_3*np.cos(i_theta)])
                    x_glob = self.rotation_mat.dot(x_loc) + self.position_center
                    x_samples.append(x_glob)
        return x_samples

    def generatePointInside(self):
        """Generate a random point inside the ellipsoid."""
        w = np.random.normal(size=3)
        # Generating 3 independent random points from the standard Gaussian distribution
        r = np.random.uniform()**(1/3)
        # Sampling the "radius"
        R = np.linalg.norm(w)
        x_loc = np.array([r*self.semi_axis_1*w[0]/R,
                          r*self.semi_axis_2*w[1]/R,
                          r*self.semi_axis_3*w[2]/R])
        x_glob = self.rotation_mat.dot(x_loc) + self.position_center
        return x_glob


    def intersectionArea(self, other_particle):
        """
        This function computes the intersection between the ellipse and the other particle.

        Parameters:
            other_particle: Particle
                Other particle
        """
        box = Particle.box
        # Saving the array defining the RVE box
        diff_in_box = self.position_center - other_particle.position_center
        diff_nearest_other = box*np.round(diff_in_box/box)
        # Computing the difference vector between the centers of the current sphere and
        # the nearest image of the other sphere
        intersection = self.intersectionEllipsoids(other_particle, diff_nearest_other)
        # Saving the class name of the other particle as a string
        if intersection:
        # There is overlap
            overlap_volume = self.intersectionVolumeEllipsoidOther(other_particle, max_it=50, seq_size=100)
            # Computing the intersection area
        else:
        # There is no overlap
            overlap_volume = 0
        return overlap_volume
        # Returning the intersection area



    def intersectionEllipsoids(self, other_ellipsoid, diff_nearest, verlet=False):
        """
        Check if the current and the other ellipsoid intersect.
        """
        def coefficientsCharacteristicEquation(M_i, axis_lengths, A_j):
            """
            Compute coefficients of the characteristic equation for ellipsoids i and j.

            Parameters
            ----------
            M_i: 4-array
            Rotation and translation matrix from local to global homogeneous coordinates of
            ellipsoid 1.

            axis_lengths: list
            List containing the length of the semi principal axes of ellipsoid 1.

            A_j: 4-array
            Global homogeneous characteristic matrix of ellipsoid j.

            Returns
            -------
            p: list
            Coefficients of the characteristic equation with p[0] the coefficient relative to
            the term of 4th order.
            """
            C = M_i.T.dot(A_j.dot(M_i))
            # Saving the auxiliar matrix C
            [a, b, c] = axis_lengths
            delta_1 = (1/a)**2
            delta_2 = (1/b)**2
            delta_3 = (1/c)**2
            # Defining the auxiliar parameters delta_1, delta_2 and delta_3
            p_1 = - delta_1*delta_2*delta_3
            p_2 = -(delta_2*delta_3*C[0, 0] + delta_1*delta_3*C[1, 1] + delta_1*delta_2*C[2, 2]
                    - delta_1*delta_2*delta_3*C[3, 3])
            p_3 = (delta_1*delta_2*(C[2, 2]*C[3, 3] - C[2,3] * C[3,2])
                   + delta_2*delta_3*(C[0, 0]*C[3, 3] - C[0, 3]*C[3,0])
                   + delta_1*delta_3*(C[1,1]*C[3,3] - C[1,3]*C[3,1])
                   + delta_1*(C[1,2]*C[2,1] - C[1,1]*C[2,2])
                   + delta_2*(C[0,2]*C[2,0] - C[0,0]*C[2,2])
                   + delta_3*(C[0,1]*C[1,0] - C[0,0]*C[1,1]))
            p_4 = (delta_1*(C[1, 1]*C[2, 2]*C[3, 3] - C[1, 1]*C[2, 3]*C[3, 2] - C[2, 2]*C[3, 1]*C[1, 3]
                   - C[3, 3]*C[2, 1]*C[1, 2] + C[2, 1]*C[1, 3]*C[3, 2] + C[3, 1]*C[1, 2]*C[2, 3])
                   + delta_2*(C[0, 0]*C[2, 2]*C[3, 3] - C[0, 0]*C[2, 3]*C[3, 2] - C[2, 2]*C[0, 3]*C[3, 0]
                   - C[3, 3]*C[0, 2]*C[2, 0] + C[2, 0]*C[0, 3]*C[3, 2] + C[3, 0]*C[0, 2]*C[2, 3])
                   + delta_3*(C[0, 0]*C[1, 1]*C[3, 3] - C[0, 0]*C[1, 3]*C[3, 1] - C[1, 1]*C[0, 3]*C[3, 0]
                   - C[3,3]*C[0,1]*C[1,0] + C[1,0]*C[0,3]*C[3,1] + C[3,0]*C[0,1]*C[1,3])
                   + C[0,0]*C[1,2]*C[2,1] + C[1,1]*C[0,2]*C[2,0] + C[2,2]*C[0,1]*C[1,0]
                   - C[0,0]*C[1,1]*C[2,2] - C[1,0]*C[0,2]*C[2,1] - C[2,0]*C[0,1]*C[1,2])
            p_5 = np.linalg.det(A_j)
            # Obtaining the coefficients
            return [p_1, p_2, p_3, p_4, p_5]

        def coefficientsEta(p_1, p_2, p_3, p_4, p_5):
            p_1_bar = p_2/(4*p_1)
            p_2_bar = p_3/(6*p_1)
            p_3_bar = -p_4/(4*p_1)
            p_4_bar = p_5/p_1

            beta_1 = (p_4_bar - p_1_bar*p_3_bar) + 3*(p_2_bar**2 - p_1_bar*p_3_bar)
            beta_2 = (-p_3_bar*(p_3_bar - p_1_bar*p_2_bar) - p_4_bar*(p_1_bar**2 - p_2_bar)
                      - p_2_bar*(p_2_bar**2 - p_1_bar*p_3_bar))

            eta_1 = beta_1**3 - 27*beta_2**2
            eta_2 = (-9*(p_3_bar - p_1_bar*p_2_bar)**2 + 27*(p_1_bar**2 - p_2_bar)
                     *(p_2_bar**2 - p_1_bar*p_3_bar) - 3*(p_4_bar - p_1_bar*p_3_bar)
                     *(p_1_bar**2 - p_2_bar))
            eta_3 = beta_1*(p_3_bar - p_1_bar*p_2_bar) - 3*p_1_bar*beta_2
            eta_4 = -(p_4_bar - p_1_bar*p_3_bar)
            eta_5 = (p_1_bar**2 - p_2_bar)
            return [eta_1, eta_2, eta_3, eta_4, eta_5]

        if verlet == True:
            [semi_axis_1, semi_axis_2, semi_axis_3] = (Particle.verlet_factor
                * np.array([self.semi_axis_1, self.semi_axis_2, self.semi_axis_3]))
        else:
            [semi_axis_1, semi_axis_2, semi_axis_3] = (
                [self.semi_axis_1, self.semi_axis_2, self.semi_axis_3])
        p = coefficientsCharacteristicEquation(self.M(), [semi_axis_1,
                                               semi_axis_2, semi_axis_3],
                                               other_ellipsoid.A_glob(verlet, diff_nearest))
        # Obtaining the coefficients of the characteristic equation
        # det(\lambda*A_i + A_j) = 0
        eta = coefficientsEta(p[0], p[1], p[2], p[3], p[4])
        # Obtaining the related coefficients eta
        cond_sep_1 = eta[0] == 0 and eta[1] >  0 and eta[2] > 0 and eta[4] > 0
        cond_sep_2 = eta[0] >  0 and eta[1] >  0                and eta[4] > 0
        cond_tan_1 = eta[0] == 0 and eta[1] >  0 and eta[2] < 0 and eta[4] > 0
        cond_tan_2 = eta[0] == 0 and eta[1] == 0 and eta[3] < 0 and eta[4] > 0
        # Computing the separation and tangent conditions from the eta coefficients
        if not(cond_sep_1 or cond_sep_2 or cond_tan_1 or cond_tan_2):
        # There is an intersetion
            intersect = True
        else:
        # There is no intersection
            intersect = False
        return intersect

    def intersectionVerlet(self, other_particle):
        '''
        This function computes the intersection between the disk and the other particle.

        Parameters:
            other_particle: Particle
                Other particle
        '''
        class_name_other_particle = other_particle.__class__.__name__
        # Saving the class name of the other particle as a string
        box = Particle.box
        # Saving the array defining the RVE box
        diff_in_box = self.position_center - other_particle.position_center
        diff_nearest_other = box*np.round(diff_in_box/box)
        # Computing the difference vector between the centers of the current sphere and
        # the nearest image of the other sphere
        intersection_verlet = self.intersectionEllipsoids(other_particle,
                                                          diff_nearest_other,
                                                          verlet=True)
        # Computing the intersection area
        return intersection_verlet
        # Returning the intersection area


    def insideVerlet(self):
        """Check if the ellipse has moved outside its Verlet neighboorhood."""
        if np.linalg.norm(self.displacement_last_verlet) >= \
            self.semi_axis_3*(Particle.verlet_factor - 1):
        # Its possible for the ellipse to have moved outside its Verlet neighboorhood
            point_in = self.pointInside(
                self.displacement_last_verlet + self.position_center, verlet=True)
            # Checking if the ellipse is still inside its Verlet neighboorhood
        else:
        # the center of the ellipse has not
            point_in = True

        return point_in

    def generatePointsOnSurface(self, n_points, erosion_thick=0):
        """Generate *n_points* on the surface of the ellipse."""
        theta = np.linspace(0, np.pi, n_points)
        phi = np.linspace(0, 2*np.pi, n_points, endpoint=False)
        # Using the convention from physics for the angles
        points_loc = []
        for i_theta in theta:
            for j_phi in phi:
                points_loc.append([
                    self.semi_axis_1*np.sin(i_theta)*np.cos(j_phi),
                    self.semi_axis_2*np.sin(i_theta)*np.sin(j_phi),
                    self.semi_axis_3*np.cos(i_theta)])
        # Generating the points in the Disk's local coordinates
        if erosion_thick > 0:
        # If erosion was sepcified
            for point_ind, i_point in enumerate(points_loc):
            # For each point on the surface with its corresponding homogeneous angle
                normal_vec = np.array([i_point[0]/self.semi_axis_1**2,
                                       i_point[1]/self.semi_axis_2**2,
                                       i_point[2]/self.semi_axis_3**2])
                unit_normal = normal_vec/np.linalg.norm(normal_vec)
                # Outward unit normal
                points_loc[point_ind] -= erosion_thick*unit_normal
                # Translation of the point in the normal direction to the surface by the
                # specified thickness (erosion)
        points_glob = np.array([self.rotation_mat.dot(point_loc) + self.position_center for point_loc in points_loc])
        # Transforming local in global coordinates
        return points_glob

    def computeCriticalErosionThickness(self):
        """Compute the critical erosion thickness for an ellipse."""
        smallest_semi_axis = np.min([self.semi_axis_1, self.semi_axis_2, self.semi_axis_3])
        largest_semi_axis = np.max([self.semi_axis_1, self.semi_axis_2, self.semi_axis_3])
        erosion_thickness = smallest_semi_axis**2/largest_semi_axis
        # Semi-latus rectum
        return erosion_thickness

class Sphere(Ellipsoid):
    '''
    This is the subclass of particles with the form of a sphere.

    Attributes
    ----------
    radius: float
        Radius of the disk
    '''
    def __init__(self, phase, radius):
        '''
        The constructor of the Sphere particle.

        Parameters
        ----------
        phase: str
            Phase to wich the particle belongs

        radius: float
            Radius of the sphere
        '''

        self.radius = radius
        super().__init__(phase, 2*radius, 2*radius, 2*radius, 0., 0., 1., 0.)

    def intersectionArea(self, other_particle):
        '''
        This function computes the intersection volume (it's called area for compatibility
        reasons) between the Sphere and the other particle.

        Parameters
        ----------
        other_particle: `.Particle`
            Other particle
        '''

        class_name_other_particle = other_particle.__class__.__name__
        # Saving the class name of the other particle as a string
        if 'Sphere'==class_name_other_particle:
        # The other particle is also a Sphere
            intersection_volume = self.intersectionVolumeSphereSphere(other_particle)
            # Computing the intersection area
            return intersection_volume
            # Returning the intersection area
        elif 'Ellipsoid' == class_name_other_particle:
        # The other particle is an Ellipsoid
            intersection_volume = other_particle.intersectionArea(self)
            # Computing the intersection area
            return intersection_volume
            # Returning the intersection area

    def intersectionVolumeSphereSphere(self, other_sphere):
        '''
        This function computes the intersection area between two Spheres.

        Parameters
        ----------
        other_sphere: `.Sphere`
        Other sphere whose intersection volume with the current sphere we want to know
        '''
        box = Particle.box
        # Saving the array defining the RVE box
        diff_center = self.position_center - other_sphere.position_center
        diff_center = diff_center - box*np.round(diff_center/box)
        # Computing the difference vector between the centers of the current sphere and
        # the nearest image of the other sphere
        d = np.linalg.norm(diff_center)
        # Distance between the current sphere and the nearest image of the other sphere
        if self.radius >= other_sphere.radius:
        # The radius of the self is larger than the radius of the other sphere
            r_1 = self.radius
            # Sphere 1 is the sphere with the larger radius
            r_2 = other_sphere.radius
            # Sphere 2 is the sphere with the smaller radius
        else:
        # The radius of the other sphere is larger than the radius of the self
            r_1 = other_sphere.radius
            # Sphere 1 is the sphere with the larger radius
            r_2 = self.radius
            # Sphere 2 is the sphere with the smaller radius
        if d >= (r_1 + r_2):
        # The spheres intersect at most at one point
            intersection_volume = 0
            # The intersection area of the spheres is zero
        elif d <= r_1 - r_2:
        # Sphere 2 is interely contained within Sphere 1
            intersection_volume = 4/3*np.pi*r_2**3
            # The intersection area is equal to the area of the smaller sphere, Sphere 2
        else:
            d_1 = (r_1**2 - r_2**2 + d**2)/(2*d)
            # x coordinate of the intersection point of the two disks if the the origin is at
            # disk 1 and the x axis goes through the center of both disks
            d_2 = d - d_1
            # Distance in the x axis from the intersection point to disk 2
            intersection_volume = (
                r_1**3/3*2*np.pi*(1 - d_1/r_1)   # Volume of spherical cap (Sphere 1)
                - d_1*(r_1**2-d_1**2)*np.pi/3    # Volume of cone (Sphere 1)
                + r_2**3/3*2*np.pi*(1 - d_2/r_2)   # Volume of shperical cap (Sphere 2)
                - d_2*(r_2**2 - d_2**2)*np.pi/3)  # Volume of cone (Sphere 2)
            # Computing the intersection area as the sum of the spherical caps minus the
            # corresponding cones
            # intersection_volume = 0.01
        return intersection_volume
        # Returning the intersection area

    def volume(self):
        
        volume = 4*np.pi/3*self.radius**3
        return volume

    def intersectionVolumeSphereEllipsoid(self, ellipse):
        pass

    def intersectionVerlet(self, other_particle):
        '''
        This function computes the intersection between the disk and the other particle.

        Parameters:
            other_particle: `.Particle`
                Other particle
        '''
        class_name_other_particle = other_particle.__class__.__name__
        # Saving the class name of the other particle as a string
        if 'Sphere' == class_name_other_particle:
        # The other particle is also a Disk
            intersection_verlet = self.intersectionVerletSphereSphere(other_particle)
            # Computing the intersection area
            return intersection_verlet
            # Returning the intersection area
        elif 'Ellipsoid' == class_name_other_particle:
            box = Particle.box
            # Saving the array defining the RVE box
            diff_in_box = self.position_center - other_particle.position_center
            diff_nearest_other = box*np.round(diff_in_box/box)
            # Computing the difference vector between the centers of the current sphere and
            # the nearest image of the other sphere
            intersection_verlet = self.intersectionEllipsoids(other_particle, diff_nearest_other, verlet=True)
            # Computing the intersection area
            return intersection_verlet
            # Returning the intersection area

    def pointInside(self, point, tol=1e-3):

        if np.linalg.norm(self.position_center-point) - self.radius <= tol:
            point_in = True
        else:
            point_in = False

        return point_in

    def intersectionVerletSphereSphere(self, other_sphere):
        """
        This function computes the intersection area between two disks
        """
        box = Particle.box
        # Saving the limits of the box
        diff_center = self.position_center - other_sphere.position_center
        diff_center = diff_center - box*np.round(diff_center/box)
        # Vector between the centers of the current disk and the nearest image of the other
        # disk
        d = np.sqrt(diff_center.dot(diff_center))
        # Distance between the disks
        if d < (self.radius + other_sphere.radius)*Particle.verlet_factor:
        # The disks are in eachothers neighboorhoods
            intersection_verlet = True
        else:
            intersection_verlet = False
        return intersection_verlet


    def insideVerlet(self):
        """Check if the ellipse has moved outside its Verlet neighboorhood."""
        if np.linalg.norm(self.displacement_last_verlet) >= \
            self.radius*(Particle.verlet_factor - 1):
        # Its possible for the ellipse to have moved outside its Verlet neighboorhood
            point_in = False
            # Checking if the ellipse is still inside its Verlet neighboorhood
        else:
        # the center of the ellipse has not
            point_in = True

    def generatePointsOnSurface(self, n_points, erosion_thick=0):
        """Generate *n_points* on the surface of the sphere."""
        theta = np.linspace(0, np.pi, n_points)
        phi = np.linspace(0, 2*np.pi, n_points, endpoint=False)
        # Convention from physics
        if erosion_thick > 0:
        # If erosion was sepcified
            radius = self.radius - erosion_thick
            # Eroding the radius
        else:
            radius = self.radius
        points_loc = []
        for i_theta in theta:
            for j_phi in phi:
                points_loc.append([
                    radius*np.sin(i_theta)*np.cos(j_phi),
                    radius*np.sin(i_theta)*np.sin(j_phi),
                    radius*np.cos(i_theta)])
        # Generating the points in the Sphere's local coordinates
        points_glob = points_loc + self.position_center
        # Transforming local in global coordinates
        print(points_glob)
        return points_glob

    def computeCriticalErosionThickness(self):
        """Compute the critical erosion thickness for a sphere."""
        erosion_thickness = 0.9*self.radius
        # Semi-latus rectum
        return erosion_thickness

def intersectionPointsEllipses(A1, B1, center_1, angle_1,
                               A2, B2, center_2, angle_2, tol=1e-10):
    """
    This function returns the y coordinates of the intersection points between two
    ellipses.

    Parameters:
        A1: float
            Semi-major axis of ellipse 1.
        B1: float
            Semi-minor axis of ellipse 1.
        center_1: array
            Coordinates of the center of ellipse 1
        angle_1: float
            Angle in radians that the major axis of ellipse 1 forms with the x-axis
        A2: float
            Semi-major axis of ellipse 2.
        B2: float
            Semi-minor axis of ellipse 2.
        center_2: array
            Coordinates of the center of ellipse 2.
        angle_2: float
            Angle in radians that the major axis of ellipse 2 forms with the x-axis

    Returns:
        intersect_points: list of arrays
            List of arrays containing the intersection points of the two ellipses
            in the original coordinate system
    """
    intersect_pts = []
    # Initializing the array containing the intersection points
    rot_mat = np.array([[ np.cos(angle_1), np.sin(angle_1)],
                        [-np.sin(angle_1), np.cos(angle_1)]])
    rot_mat_back = rot_mat.T
    # Rotation matrix that alignes ellipse 1 with the xy-axis
    center_2_TR = rot_mat.dot(center_2 - center_1)
    # Translation and rotation of ellipse 2 with the origin at the center of ellipse 1
    # aligning with the xy axis
    theta = angle_2 - angle_1
    # Saving the angle between the axis of both ellipses
    # AA = np.cos(theta)**2/A2**2 + np.sin(theta)**2/B2**2
    # BB = 2*np.sin(theta)*np.cos(theta)/A2**2-2*np.sin(theta)*np.cos(theta)/B2**2
    # CC = np.sin(theta)**2/A2**2+np.cos(theta)**2/B2**2
    # DD = -2*np.cos(theta)*(center_2_TR[0]*np.cos(theta)+\
    #     center_2_TR[1]*np.sin(theta))/A2**2 +\
    #     2*np.sin(theta)*(center_2_TR[1]*np.cos(theta)-\
    #     center_2_TR[0]*np.sin(theta))/B2**2
    # EE = -2*np.sin(theta)*(center_2_TR[0]*np.cos(theta)+\
    #     center_2_TR[1]*np.sin(theta))/A2**2 +\
    #     2*np.cos(theta)*(center_2_TR[0]*np.sin(theta)-\
    #     center_2_TR[1]*np.cos(theta))/B2**2
    # FF = (center_2_TR[0]*np.cos(theta)+center_2_TR[1]*np.sin(theta))**2/A2**2+\
    #     (center_2_TR[0]*np.sin(theta)-center_2_TR[1]*np.cos(theta))**2/B2**2 - 1
    AA = A2**2*np.sin(theta)**2 + B2**2*np.cos(theta)**2
    BB = 2*(B2**2-A2**2)*np.sin(theta)*np.cos(theta)
    CC = A2**2*np.cos(theta)**2 + B2**2*np.sin(theta)**2
    DD = -2*AA*center_2_TR[0] - BB*center_2_TR[1]
    EE = -BB*center_2_TR[0] - 2*CC*center_2_TR[1]
    FF = AA*center_2_TR[0]**2 + BB*center_2_TR[0]*center_2_TR[1] + CC*center_2_TR[1]**2 -\
        A2**2*B2**2
    # Coefficients defining ellipse 2 on the coordinate system of ellipse 1
    # AA*x**2+BB*x*y+CC*y**2+DD*x+EE*y+FF=0
    # from sympy import var, plot_implicit, Eq
    # var('x y')
    # plot_implicit(Eq(AA*x**2+BB*x*y+CC*y**2+DD*x+EE*y+FF,0))
    p = np.zeros(5)
    # Initializing the vector of the coefficients
    # p[0] = A1**4*AA**2 + B1**2*(A1**2*(BB**2-2*AA*CC)+B1**2*CC**2)
    # p[1] = 2*B1*(B1**2*CC*EE+A1**2*(BB*DD-AA*EE))
    # p[2] = A1**2*((B1**2*(2*AA*CC-BB**2)+DD**2-2*AA*FF)-2*A1**2*AA**2)+\
    #     B1**2*(2*CC*FF+EE**2)
    # p[3] = 2*B1*(A1**2*(AA*EE-BB*DD)+EE*FF)
    # p[4] = ((A1*(A1*AA-DD)+FF)*(A1*(A1*AA+DD)+FF))
    # p[0] = -CC**2*B1**4 + 2*(AA*CC - BB**2/2)*A1**2*B1**2 - A1**4*AA**2
    # p[1] = (2*(-AA*CC + BB**2/2)*A1**2 - 2*CC*FF - EE**2)*B1**4 + 2*(AA**2*A1**2 + AA*FF - \
    #     1/2*DD**2)*A1**2*B1**2
    # p[2] = -2*B1**4*CC*EE + (2*A1**2*AA*EE - 2*A1**2*BB*DD)*B1**2
    # p[3] = -(A1**2*AA - A1*DD + FF)*(A1**2*AA + A1*DD + FF)*B1**4
    # p[4] = -((A1**2*AA - A1*DD + FF)*(A1*BB + EE) + (-A1*BB + EE)*(A1**2*AA + A1*DD + FF))*B1**4
    p[0] = -CC**2*B1**4 + 2*(AA*CC - BB**2/2)*A1**2*B1**2 - A1**4*AA**2
    p[1] = -((-A1*BB + EE)*CC + CC*(A1*BB + EE))*B1**4 + 2*(AA*EE - BB*DD)*A1**2*B1**2
    p[2] = -((A1**2*AA - A1*DD + FF)*CC + (-A1*BB + EE)*(A1*BB + EE) + CC*(A1**2*AA + A1*DD + FF))*B1**4 + 2*(AA**2*A1**2 + AA*FF - 1/2*DD**2)*A1**2*B1**2
    p[3] = -((A1**2*AA - A1*DD + FF)*(A1*BB + EE) + (-A1*BB + EE)*(A1**2*AA + A1*DD + FF))*B1**4
    p[4] = -(A1**2*AA - A1*DD + FF)*(A1**2*AA + A1*DD + FF)*B1**4
    # Coefficients of the polynomial expressing the intersection of the two ellipses
    y_pts = []
    roots = set(np.roots(p))
    # Roots of the polynomial, with positive values giving the y values of the
    # intersection points in the coordinate system of ellipse 1
    for i_root in roots:
    # Running through all the roots
        if np.abs(np.imag(i_root))<tol:
        # if the root is real, then it is the y-coordinate of an intersection point
            y_pt = np.real(i_root)
            if not np.any(np.isclose(y_pt*np.ones(len(y_pts)),y_pts)) or len(y_pts)==0:
                y_pts.append(y_pt)
                on_ellipse_2 = False
                x_pt = A1*np.sqrt(1-y_pt**2/B1**2)
                # (x_pt, y_pt) and (-x_pt,y_pt) are the coordinates of the potential
                # intersection points obtained assuming that they are on ellipse 1
                on_ellipse_2_1 = \
                    np.abs(AA*x_pt**2 + BB*x_pt*y_pt + CC*y_pt**2 + DD*x_pt + EE*y_pt + FF) < tol
                # Checking if (x_pt, y_pt) is also on ellispe 2 and so it's a real
                # intersection point
                on_ellipse_2_2 = \
                    np.abs(AA*x_pt**2 - BB*x_pt*y_pt + CC*y_pt**2 - DD*x_pt + EE*y_pt + FF) < tol
                # Checking if (-x_pt, y_pt) is also on ellispe 2 and so it's a real
                # intersection point
                if on_ellipse_2_1:
                # (x_pt, y_pt) is a true intersection point
                    intersect_pts.append(rot_mat_back.dot(np.array([x_pt, y_pt]))+center_1)
                    # Append the point to the list of intersection points in the original
                    # coordinate system
                if on_ellipse_2_2:
                # (-x_pt, y_pt) is a true intersectio point
                    intersect_pts.append(rot_mat_back.dot(np.array([-x_pt, y_pt]))+center_1)
                    # Append the point to the list of intersection points in the original
                    # coordinate system
                # if on_ellipse_2_1 and on_ellipse_2_2 and np.abs(x_pt)<0.05:
                #     intersect_pts.pop()
    return intersect_pts


def uniformSampleEllipse(center, A, B, angle):

    z = np.array([0., 0.])
    z[0] = np.random.normal()
    z[1] = np.random.normal()
    r = np.random.uniform()**(1/2)
    R = np.linalg.norm(z)
    x = r*A*z[0]/R
    y = r*B*z[1]/R
    rot_mat = np.array([[ np.cos(angle), -np.sin(angle)],
                        [np.sin(angle), np.cos(angle)]])
    [x, y] = rot_mat.dot([x, y])

    return [x + center[0] , y + center[1]]

def regularSampleEllipse(center, A, B, angle, n_samples):
    n_theta = int(np.sqrt(n_samples**(1)))
    n_r = int(np.round(n_samples/n_theta))
    print(n_theta, n_r)
    theta = np.linspace(0, 2*np.pi, n_theta, endpoint=False)
    rot_mat = np.array([[ np.cos(angle), -np.sin(angle)],
                        [np.sin(angle), np.cos(angle)]])
    r = np.linspace(0.01, 1, n_r, endpoint=True)
    k_sample = 0
    x = []
    y = []
    for i_theta in theta:
        for j_radius in r:
            x.append(j_radius*A*np.cos(i_theta))
            y.append(j_radius*B*np.sin(i_theta))
            
            [x[k_sample], y[k_sample]] = rot_mat.dot([x[k_sample], y[k_sample]]) + [center[0], center[1]]
            k_sample += 1

    return [x, y] 



if __name__ == '__main__':
# Test drive

    
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from scipy import integrate
    import time


    Particle.volume = 0
    Particle.number = 0
    Particle.box = [1., 1., 1.]
    
    
    
    ellipsoid_1 = Ellipsoid('1', 0.3, 0.3, 0.2, np.sqrt(3)/3, np.sqrt(3)/3, np.sqrt(3)/3, 0)
    ellipsoid_1.position_center = np.array([0.95, 0.5, 0.5])
    ellipsoid_2 = Ellipsoid('1', 0.3, 0.3, 0.3, 0., 0., 1., 0)
    ellipsoid_2.position_center = np.array([0.05, 0.5, 0.6])
    
    box = Particle.box
    # Saving the array defining the RVE box
    diff_in_box = ellipsoid_1.position_center - ellipsoid_2.position_center
    diff_nearest_other = box*np.round(diff_in_box/box)
    # Computing the difference vector between the centers of the current sphere and
    # the nearest image of the other sphere
    
    intersect = ellipsoid_1.intersectionEllipsoids(ellipsoid_2, diff_nearest_other)
    print(intersect)
    
    start_1 = time.time()
    overlap_volume_1 = ellipsoid_1.intersectionVolumeEllipsoidOther(ellipsoid_2, type='random')
    end_1 = time.time()
    start_2 = time.time()
    overlap_volume_2 = ellipsoid_1.intersectionVolumeEllipsoidOther(ellipsoid_2, type='regular')
    end_2 = time.time()
    v_ellipsoid_2 = ellipsoid_2.volume()
    print(overlap_volume_1, end_1-start_1, overlap_volume_2, end_2-start_2)

    # Particle.volume = 0
    # Particle.number = 0
    # Particle.box = [1., 1.]
    # 
    # ellipse_1 = Ellipse('1', 0.4, 0.2, 0)
    # ellipse_1.position_center = np.array([0.6, 0.5])
    # ellipse_2 = Ellipse('1', 0.4, 0.2, np.pi/3)
    # ellipse_2.position_center = np.array([0.6, 0.5])
    # 
    # particles = [ellipse_1, ellipse_2]
    # fig = plt.figure()
    # 
    # ax = plt.gca()
    # 
    # N = len(particles)
    # 
    # box = Particle.box
    # # Saving the RVE dimensions
    # diff_in_box = ellipse_1.position_center - ellipse_2.position_center
    # # Difference vector between the center of the two ellipses
    # diff_nearest_other = box*np.round(diff_in_box/box)
    # # Vector from the position of the other ellipse to its nearest image to the current
    # # ellipse
    # 
    # intersect_pts = np.array(intersectionPointsEllipses(ellipse_1.semi_major_axis, ellipse_1.semi_minor_axis,
    #     ellipse_1.position_center, ellipse_1.angle, ellipse_2.semi_major_axis,
    #     ellipse_2.semi_minor_axis, ellipse_2.position_center + diff_nearest_other, ellipse_2.angle))
    # 
    # intersect_pts_ord = ellipse_1.sortPointsOnEllipse(intersect_pts)
    # 
    # 
    # for i in range(N):
    #     for j in range(-1,2):
    #         for k in range(-1,2):
    #             ellip = mpatches.Ellipse(particles[i].position_center+np.array([1*j,1*k]), particles[i].major_axis, particles[i].minor_axis,angle=180/np.pi*particles[i].angle,alpha=0.1)
    #             ax.add_artist(ellip)
    #             plt.annotate(xy = particles[i].position_center, s=str(i))
    #             plt.scatter(particles[i].position_center[0],particles[i].position_center[1])
    #             plt.axis([0, 1, 0, 1])
    # 
    # number = 20
    # k = 0
    # for i_point in range(number):
    #     [x, y] = uniformSampleEllipse(ellipse_1.position_center, ellipse_1.semi_major_axis, ellipse_1.semi_minor_axis, ellipse_1.angle)
    #     point_in = ellipse_2.pointInside(np.array([x, y])-diff_nearest_other)
    #     if point_in:
    #         plt.scatter(x, y, c='r', s=1)
    #         k += 1
    #     else:
    #         plt.scatter(x, y, c='k', s=1)
    # 
    # x, y = regularSampleEllipse(ellipse_1.position_center, ellipse_1.semi_major_axis, ellipse_1.semi_minor_axis, ellipse_1.angle, number)
    # k_reg = 0
    # for i_point in range(len(x)):
    #     point_in = ellipse_2.pointInside(np.array([x[i_point], y[i_point]])-diff_nearest_other)
    #     if point_in:
    #         plt.scatter(x[i_point], y[i_point], c='b', s=1)
    #         k_reg += 1
    #     else:
    #         plt.scatter(x[i_point], y[i_point], c='g', s=1)
    # 
    # 
    # A = ellipse_1.semi_major_axis
    # B = ellipse_1.semi_minor_axis
    # def pointsInside(x, y):
    #     [x_glob, y_glob] = ellipse_1.rot_mat.dot([x, y]) + ellipse_1.position_center
    #     pointIn = ellipse_2.pointInside(ellipse_1.rot_mat.dot([x, y]) + ellipse_1.position_center)
    #     if pointIn:
    #         value = 1
    #     else:
    #         value = 0
    #     return value
    # 
    # A1 = ellipse_1.intersectionArea(ellipse_2)
    # print('exact', A1)
    # A2 = ellipse_1.volume()*k/number
    # print('approx', A2)
    # A3 = ellipse_1.volume()*k_reg/number
    # print('approx_reg', A3)
    # A4 = integrate.dblquad(pointsInside, -B, B, lambda y: -A*np.sqrt(1 - y**2/B**2), lambda y: A*np.sqrt(1 - y**2/B**2), epsrel=1 )
    # print('quad', A4[0])
    # 
    # for i_intr_pt in range(len(intersect_pts_ord)):
    #     midpoint = ellipse_1.midpointOnEllipse(intersect_pts_ord[i_intr_pt], intersect_pts_ord[np.mod(i_intr_pt+1,len(intersect_pts_ord))])
    #     plt.scatter(midpoint[0], midpoint[1], color='r')
    # 
    # intersect_pts_ord = np.array(intersect_pts_ord)
    # plt.scatter(intersect_pts_ord[:,0], intersect_pts_ord[:,1])
    # for i_intr_pt in range(len(intersect_pts_ord)):
    #     plt.annotate(xy = intersect_pts_ord[i_intr_pt,:], s=str(i_intr_pt))
    # 
    # 
    # # plt.axis([-1, 2, -1, 2])
    # plt.show()
