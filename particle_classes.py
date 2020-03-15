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

    def __init__(self, dim):
        '''
        The constructor for the Particle class.

        Parameters:
            dim: int
                Number of the dimensions of the space where the particle "lives"
        '''

        self.dim = dim
        self.force = np.zeros((dim))
       # Setting the the dimension where the particle "lives"

    def setPositionCenter(self, position):
        '''
        This function sets the position of the center of mass of the particle
        '''

        self.position_center = position_center
        # Setting the position of the center of mass of the particle

    def setVelocityCenter(self, velocity):
        '''
        This function sets the velocity of the center of mass of the particle.
        '''

        self.velocity_center = velocity_center
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

        box = np.array([1, 1],dtype='float')

        vector_centers = other_particle.position_center - self.position_center #+ np.random.uniform(low=-0.2,high=0.2,size=2)
        vector_centers = vector_centers - box*np.round(vector_centers/box)
        if np.linalg.norm(vector_centers) != 0:
            unit_vector_i_j = vector_centers/np.linalg.norm(vector_centers)
        else:
            random_vector = np.random.uniform(size=self.dim)
            unit_vector_i_j = random_vector/np.linalg.norm(random_vector)
        return unit_vector_i_j


# ==========================================================================================
class Disk(Particle):
    '''
    This is the subclass of particles with the form of a circular disk.

    Attributes:
        radius: int
            Radius of the disk
    '''
    def __init__(self, radius):
        '''
        The constructor of the Disk particle.

        Parameters:
            center: list
                The position vector of the center of mass of the particle
            dim: int
                Number of the dimensions of the space where the particle "lives"
            radius: float
                Radius of the disk
        '''
        self.dim = 2
        self.radius = radius
        self.force = np.zeros((self.dim),dtype='float')

    def intersectionArea(self, other_particle):
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
            intersection_area = self.intersectionAreaDiskDisk(other_particle)
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
