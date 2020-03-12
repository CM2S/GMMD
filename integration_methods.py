def Newmark(x_0, x_dot_0, f_vec, m_mat, c_mat, k_mat, dt, n_steps, dim):
    '''
    This function integrates the equations of motion using Newmark's method.

    Parameters:
        x_0: vector array
            Initial positions of the DOFs
        x_dot_0: vector array
            Initial velocities of the DOFs
        f_vec: vector array
            Forves acting on the DOFs at each time instant
        m_mat: matrix array
            Mass matrix
        c_mat: matrix array
            Damping matrix
        k_mat: matrix array
            Stiffness matrix
        dt: float
            Time step
        n_steps: int
            Number of time steps to be used
        dim: int
            Dimension of the problem

    Returns:
        x_vec: vector array
            Positions
        x_dot_vec: vector array
            Velocities
        x_ddot_vec: vector array
            Acceleration
    '''
    x_vec = np.zeros(dim, n_steps+1)
    x_vec[:,0] = x_0
    x_dot_vec = np.zeros(dim, n_steps+1)
    x_dot_vec[:,0] = x_dot_0
    x_ddot_vec = np.zeros(dim, n_steps+1)
    # Initializing the array vectors containing the positions, velocities and accelerations
    x_ddot_vec[:,0] = \
        numpy.linalg.solve(m_mat, f_vec[:,0] - c_mat.dot(x_dot_0) - k_mat.dot(x_0))
    # Computing the accelaration at time instant 0
    delta = 0.5
    alpha = 0.25
    a_0 = 1/(alpha*dt**2)
    a_1 = delta/(alpha*dt)
    a_2 = 1/(alpha*dt)
    a_3 = 1/(2*alpha) - 1
    a_4 = delta/alpha - 1
    a_5 = dt/2*(delta/alpha - 2)
    a_6 = dt*(1 - delta)
    a_7 = delta*dt
    # Computing the constants used in the integration algorithm
    k_mat_eff = k_mat + a_0*m_mat + a_1*c_mat
    # Computing the effective stiffness matrix
    step = 0
    # Initializing the step counter
    while step<n_steps:
        # Repeat n_steps times
        f_vec_eff = f_vec + m_mat.dot(a_0*x_vec[:,step] + a_2*x_dot_vec[:,step] + \
            a_3*x_ddot_vec[:,step]) + c_mat.dot(a_1*x_vec[:,step] + a_4*x_dot_vec[:,step] +\
            a_5*x_ddot_vec[:,step])
        # Computing the effective force at time step*dt
        x_vec[:,step+1] = numpy.linalg.solve(k_mat_eff, f_vec_eff)
        # Computing the position vector at time (step+1)*dt
        x_ddot_vec[:,step+1] = a_0*(x_vec[:,step+1] - x_vec[:,step]) - \
            a_2*x_dot_vec[:,step] - a_3*x_ddot[:,step]
        # Computing the acceleration vector at time (step+1)*dt
        x_dot_vec[:,step+1] = x_dot_vec[:,step] + a_6*x_ddot_vec[:,step] + \
            a_7*x_ddot_vec[:,step+1]
        # Computing the velocity vector at time (step+1)*dt
return [x_vec[:,1:], x_dot_vec[:,1:], x_ddot_vec[:,1:]]
