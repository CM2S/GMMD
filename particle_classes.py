import numpy as np

class Particle():
    '''
    This is the class for particles

    Attributes:
        center: list
            The position vector of the center of mass of the particle
        dim: int
            Number of the dimensions of the space where the particle "lives"
    '''

    def __init__(self, dim, phase):
        '''
        The constructor for the Particle class.

        Parameters:
            dim: int
                Number of the dimensions of the space where the particle "lives"
            phase: string
                Phase to which the particle belongs
        '''

        self.dim = dim
        self.force = np.zeros((dim))
        self.n_cell_dim = []
       # Setting the the dimension where the particle "lives"
        self.verlet_list = []
        Particle.volume += self.volume()
        Particle.number += 1

        self.phase = phase

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

    def intersectionVector(self, other_particle):
        '''
        This function computes the unit vector from the center of masss of particle i to
        particle j
        '''

        box = Particle.box

        vector_centers = other_particle.position_center - self.position_center
        vector_centers = vector_centers - box*np.round(vector_centers/box)
        # Vector connecting the centers of the current particle and the nearest image of
        # the other particle
        if self.dim==2:
            angle_opposite = np.arctan2(vector_centers[1],vector_centers[0])
            if np.random.uniform() > 0:
                angle_new = angle_opposite + np.random.uniform(low=-np.pi/4,high=np.pi/4)
            else:
                angle_new = angle_opposite
            if np.linalg.norm(vector_centers) != 0:
                unit_vector_i_j = np.array([np.cos(angle_new), np.sin(angle_new)])
                # unit_vector_i_j = vector_centers/np.linalg.norm(vector_centers)
            else:
                random_vector = np.random.uniform(size=self.dim)
                unit_vector_i_j = random_vector/np.linalg.norm(random_vector)
            return unit_vector_i_j
        elif self.dim==3:
            if np.linalg.norm(vector_centers) != 0:
                unit_vector_i_j = vector_centers/np.linalg.norm(vector_centers)
                # unit_vector_i_j = vector_centers/np.linalg.norm(vector_centers)
            else:
                random_vector = np.random.uniform(size=self.dim)
                unit_vector_i_j = random_vector/np.linalg.norm(random_vector)
            return unit_vector_i_j

# ==========================================================================================


class Disk(Particle):
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
        self.force = np.zeros((self.dim), dtype='float')
        self.n_cell_dim = []
        self.verlet_list = []
        Particle.volume += self.volume()
        Particle.number += 1
        self.phase = phase

    def intersectionArea(self, other_particle):
        '''
        This function computes the intersection between the disk and the other particle.

        Parameters:
            other_particle: Particle
                Other particle
        '''
        class_name_other_particle = other_particle.__class__.__name__
        # Saving the class name of the other particle as a string
        if 'Disk' == class_name_other_particle:
        # The other particle is also a Disk
            intersection_area = self.intersectionAreaDiskDisk(other_particle)
            # Computing the intersection area
            return intersection_area
            # Returning the intersection area
        elif 'Ellipse'==class_name_other_particle:
        # The other particle is an Ellipse
            intersection_area = self.intersectionAreaDiskEllipse(other_particle)
            # Computing the intersection area
            return intersection_area
            # Returning the intersection area

    def intersectionAreaDiskDisk(self, other_disk):
        '''
        This function computes the intersection area between two disks
        '''
        box = np.array([1,1],dtype='float')


        diff_center = self.position_center - other_disk.position_center

        diff_center = diff_center - box*np.round(diff_center/box)

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
        if d>=(r_1 + r_2):
        # The disks intersect at most at one point
            intersection_area = 0
            # The intersection area of the disks is zero
        elif d<=r_1 - r_2:
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

    def intersectionAreaDiskEllipse(self, ellipse):
        pass

    def intersectionVerlet(self, other_particle):
        '''
        This function computes the intersection between the disk and the other particle.

        Parameters:
            other_particle: Particle
                Other particle
        '''
        class_name_other_particle = other_particle.__class__.__name__
        # Saving the class name of the other particle as a string
        if 'Disk'==class_name_other_particle:
        # The other particle is also a Disk
            intersection_verlet = self.intersectionVerletDiskDisk(other_particle)
            # Computing the intersection area
            return intersection_verlet
            # Returning the intersection area
    def pointInside(self, point):
        
        if np.linalg.norm(self.position_center-point)<=self.radius:
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
        if d<(self.radius+other_disk.radius)*Particle.verlet_factor:
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

class Sphere(Particle):
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
        super().__init__(3, phase)

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
        elif 'Ellipsoid'==class_name_other_particle:
        # The other particle is an Ellipsoid
            intersection_volume = self.intersectionVolumeSphereEllipsoid(other_particle)
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
            # d_1 = (r_1**2 - r_2**2 + d**2)/(2*d)
            # # x coordinate of the intersection point of the two disks if the the origin is at
            # # disk 1 and the x axis goes through the center of both disks
            # d_2 = d - d_1
            # # Distance in the x axis from the intersection point to disk 2
            # intersection_volume = (
            #     r_1**3/3*2*np.pi*(1 - d_1/r_1)   # Volume of spherical cap (Sphere 1)
            #     - d_1*(r_1**2-d_1**2)*np.pi/3    # Volume of cone (Sphere 1)
            #     + r_2**3/3*2*np.pi*(1 - d_2/r_2)   # Volume of shperical cap (Sphere 2)
            #     - d_2*(r_2**2 - d_2**2)*np.pi/3)  # Volume of cone (Sphere 2)
            # # Computing the intersection area as the sum of the spherical caps minus the
            # # corresponding cones
            intersection_volume = 0.01
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

    def pointInside(self, point):
        
        if np.linalg.norm(self.position_center-point)<=self.radius:
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

class Ellipse(Particle):
    """docstring for Ellipse."""

    def __init__(self, phase, major_axis, minor_axis, angle):
        '''
        This is the generator for the classe Ellipse.

        Parameters:
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
        super().__init__(2, phase)

    def volume(self):
        '''
        This function computes the area(volume) of the ellipse.
        '''

        volume = np.pi*self.semi_major_axis*self.semi_minor_axis

        return volume

    def pointInsideEllipse(self, point, tol=1e-4, position='inside', verlet=False):
        '''
        This function determines if the point is inside, outside or on the ellipse given a tolerance

        Parameters:
            self: Ellipse
                Ellipse under analysis
            point: array
                Point under analysis
            tol: float
                Tolerance
            position: string
                'inside' or 'on'
            verlet: boolean
                Inside the ellipse itself or its neighboor, related to the Verlet list

        Returns:
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
            semi_minor_axis = self.semi_minor_axis*Particle.verlet_factor
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

    def pointInside(self, point):

        point_in = self.pointInsideEllipse(point)

        return point_in

    def intersectionAreaEllipseEllipse(self, other_ellipse):
        '''
        This function computes the intersection area between two ellipses.
        '''

        box = Particle.box


        diff_in_box = self.position_center - other_ellipse.position_center
        # Difference vector between the center of the two ellipses
        diff_nearest_other = box*np.round(diff_in_box/box)
        intersect_pts = intersectionPointsEllipses(
            self.major_axis/2, self.minor_axis/2,  self.position_center, self.angle,
            other_ellipse.major_axis/2, other_ellipse.minor_axis/2,
            other_ellipse.position_center + diff_nearest_other, other_ellipse.angle)
        # Computing the intersection points of the two ellipses
        if len(intersect_pts)==0:
        # Either the ellipses are disjoint or one of them is completly inside the other
            if self.volume() >= other_ellipse.volume():
            # The current ellipse is larger than the other ellipse
                if self.pointInsideEllipse(other_ellipse.position_center):
                # The other ellipse is completly inside the current ellipse
                    intersection_area = other_ellipse.volume()
                    # The intersection area is the area of the smaller ellipse
                else:
                # The ellipses are disjoint
                    intersection_area = 0
                    # The intersection area is 0
            else:
                if other_ellipse.pointInsideEllipse(self.position_center):
                # The current ellipse is completly inside the other ellipse
                    intersection_area = self.volume()
                    # The intersection area is the area of the smaller ellipse
                else:
                # The ellipses are disjoint
                    intersection_area = 0
                    # The intersection area is 0
        elif len(intersect_pts)==1:
        # Either the ellipses are disjoint or one of them is completly inside the other,
        # except for the intersection point
            if self.volume() >= other_ellipse.volume():
            # The current ellipse is larger than the other ellipse
                if self.pointInsideEllipse(other_ellipse.position_center):
                # The other ellipse is completly inside the current ellipse
                    intersection_area = other_ellipse.volume()
                    # The intersection area is the area of the smaller ellipse
                else:
                # The ellipses are disjoint
                    intersection_area = 0
                    # The intersection area is 0
            else:
                if other_ellipse.pointInsideEllipse(self.position_center):
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
            if other_ellipse.pointInsideEllipse(midpoint):
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
            for i_segment in range(1,2):
            # Running through each segment
                k_ellipse = np.mod(k_ellipse+1,2)
                # Index of the ellipse whose area segment needs to calculated
                intersection_area += \
                    ellipses[k_ellipse].areaEllipseSection(
                        intersect_pts_ord[np.mod(i_segment,2)] - (k_ellipse-1)*diff_nearest_other, \
                        intersect_pts_ord[np.mod(i_segment+1,2)] - (k_ellipse-1)*diff_nearest_other)
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
                (intersect_pts_ord[2][0]-intersect_pts_ord[0][0])*\
                (intersect_pts_ord[3][1]-intersect_pts_ord[1][1])-\
                (intersect_pts_ord[3][0]-intersect_pts_ord[1][0])*\
                (intersect_pts_ord[2][1]-intersect_pts_ord[0][1]))
            # Computing the area of the quadrilateral inscribed in the overlap of the
            # two ellipses
            ellipses = [self, other_ellipse]
            midpoint = self.midpointOnEllipse(intersect_pts_ord[0], intersect_pts_ord[1])
            if other_ellipse.pointInsideEllipse(midpoint):
                intersection_area += \
                    self.areaEllipseSection(intersect_pts_ord[0], intersect_pts_ord[1])
                k_ellipse = 1
            else:
                intersection_area += \
                    other_ellipse.areaEllipseSection(
                        intersect_pts_ord[0] - diff_nearest_other,
                        intersect_pts_ord[1] - diff_nearest_other)
                k_ellipse = 0
            for i_segment in range(1,4):
            # Running through each segment
                k_ellipse = np.mod(k_ellipse+1,2)
                intersection_area += \
                    ellipses[k_ellipse].areaEllipseSection(
                        intersect_pts_ord[np.mod(i_segment,4)] - (k_ellipse-1)*diff_nearest_other, \
                        intersect_pts_ord[np.mod(i_segment+1,4)] - (k_ellipse-1)*diff_nearest_other)
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
        midpoint = self.position_center + rot_mat_back.dot(midpoint_loc)
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
        class_name_other_particle = other_particle.__class__.__name__
        # Saving the class name of the other particle as a string
        if 'Disk'==class_name_other_particle:
        # The other particle is a Disk
            intersection_area = self.intersectionAreaEllipseDisk(other_particle)
            # Computing the intersection area
            return intersection_area
            # Returning the intersection area
        elif 'Ellipse'==class_name_other_particle:
        # The other particle is an Ellipse
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
        class_name_other_particle = other_particle.__class__.__name__
        # Saving the class name of the other particle as a string
        if 'Ellipse'==class_name_other_particle:
        # The other particle is also a Disk
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
                if self.pointInsideEllipse(other_ellipse.position_center):
                # The other ellipse is completly inside the current ellipse
                    intersection_verlet = True
                    # The intersection area is the area of the smaller ellipse
                else:
                # The ellipses are disjoint
                    intersection_verlet = False
                    # The intersection area is 0
            else:
                if other_ellipse.pointInsideEllipse(self.position_center):
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
            point_in = self.pointInsideEllipse(
                self.displacement_last_verlet + self.position_center, verlet=True)
            # Checking if the ellipse is still inside its Verlet neighboorhood
        else:
        # the center of the ellipse has not
            point_in = True

        return point_in

def intersectionPointsEllipses(A1, B1, center_1, angle_1,
    A2, B2, center_2, angle_2, tol=1e-10):
    '''
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
    '''
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

if __name__ == '__main__':
# Test drive
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    Particle.volume = 0
    Particle.number = 0
    Particle.box = [1., 1.]

    ellipse_1 = Ellipse(0.4,0.2,0)
    ellipse_1.position_center = np.array([0.02315812, 0.72524039])
    ellipse_2 = Ellipse(0.4,0.2,np.pi/2)
    ellipse_2.position_center =  np.array([0.10147887, 0.75142281])

    intersect_pts = np.array(intersectionPointsEllipses(ellipse_1.semi_major_axis, ellipse_1.semi_minor_axis,
        ellipse_1.position_center, ellipse_1.angle, ellipse_2.semi_major_axis,
        ellipse_2.semi_minor_axis, ellipse_2.position_center, ellipse_2.angle))


    particles = [ellipse_1, ellipse_2]
    fig = plt.figure()
    
    ax = plt.gca()
    
    N = len(particles)
    
    for i in range(N):
        for j in range(-1,2):
            for k in range(-1,2):
                ellip = mpatches.Ellipse(particles[i].position_center+np.array([1*j,1*k]), particles[i].major_axis, particles[i].minor_axis,angle=180/np.pi*particles[i].angle,alpha=0.8)
                ax.add_artist(ellip)
                plt.annotate(xy = particles[i].position_center, s=str(i))
                plt.scatter(particles[i].position_center[0],particles[i].position_center[1])
                plt.axis([0, 1, 0, 1])
    

    plt.scatter(intersect_pts[:,0],intersect_pts[:,1])
    plt.annotate(xy = intersect_pts[0,:], s=str(0))
    plt.annotate(xy = intersect_pts[1,:], s=str(1))

    plt.show()
