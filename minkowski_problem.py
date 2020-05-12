import sympy as sp
import numpy as np
from sympy.integrals.quadrature import gauss_legendre

def diff_operator_A(expr_func):
    theta, phi = sp.symbols('theta phi')
    A = sp.diff(expr_func, phi, 2) + expr_func
    return A

def diff_operator_B(expr_func):
    theta, phi = sp.symbols('theta phi')
    B = 1/sp.sin(phi)*sp.diff(expr_func, phi, theta) - sp.cos(phi)/(sp.sin(phi)**2)*sp.diff(expr_func, theta)
    return B

def diff_operator_C(expr_func):
    theta, phi = sp.symbols('theta phi')
    C = 1/(sp.sin(phi)**2)*sp.diff(expr_func, theta, 2) + sp.cos(phi)/sp.sin(phi)*sp.diff(expr_func, phi) + expr_func
    return C
    
def generateExprRealSphHarmonics(flat_ind, **kwargs):
    """
    Generate the expression for the real spherical harmonic with flat index *flat_ind*.

    Parameters:
    flat_ind: index of the spherical harmonic running in increasing order and degree, such
    that:

        degree = np.floor(np.sqrt(flat_ind))
        order = flat_ind - degree**2 - degree
    """
    theta, phi = sp.symbols('theta phi')
    # symbols theta and phi (not physics convention)
    l_degree = np.floor(np.sqrt(flat_ind))
    m_order = flat_ind - l_degree**2 - l_degree
    # Computing the  order and the degree of the spherical harmonic
    if m_order < 0:
        real_sph_expr = sp.simplify(1j*sp.sqrt(1/2)*(sp.simplify(sp.Ynm(l_degree, m_order, phi, theta).expand(func=True))
                                    - (-1)**m_order*sp.simplify(sp.Ynm(l_degree, -m_order, phi, theta).expand(func=True))).expand(func=True))
    elif m_order == 0:
        real_sph_expr = sp.simplify(sp.Ynm(l_degree, m_order, phi, theta).expand(func=True))
    elif m_order > 0:
        real_sph_expr = sp.simplify(sp.sqrt(1/2)*(sp.simplify(sp.Ynm(l_degree, -m_order, phi, theta).expand(func=True))
                                    + (-1)**m_order*sp.simplify(sp.Ynm(l_degree, m_order, phi, theta).expand(func=True))).expand(func=True))
    print(real_sph_expr)
    return real_sph_expr

def gradSphericalCoords(expr):
    """Gradient in unitary spherical coordinates."""
    theta, phi = sp.symbols("theta phi")
    grad_expr = np.array([sp.diff(sp.simplify(sp.expand_func(expr).expand(func=True)), phi), sp.simplify(expr.expand(func=True)).diff(theta)])
    return grad_expr


def hessSphericalCoords(expr):
    theta, phi = sp.symbols("theta phi")
    hess_expr = np.array([
        [sp.diff(expr, phi, 2), sp.diff(1/sp.sin(phi)*sp.diff(expr, theta), phi)],
        [1/sp.sin(phi)*sp.diff()]
])

def realCoeffsToComplex(c, degree):
    """Convert real coefficients to imaginary coefficients for spherical harmonics."""
    flat_ind = 0
    complex_coeffs = np.zeros((2*int(degree) + 1), dtype=complex)
    for m_order in range(-degree, degree + 1):
        flat_ind_m_plus = m_order + degree
        flat_ind_m_minus = -m_order + degree
        if m_order < 0:
            complex_coeffs[flat_ind] = \
                1 / np.sqrt(2)*(c[flat_ind_m_minus] - 1j*c[flat_ind_m_plus])
        elif m_order == 0:
            complex_coeffs[flat_ind] = c[flat_ind_m_plus]
        elif m_order > 0:
            complex_coeffs[flat_ind] = \
                1 / np.sqrt(2)*(c[flat_ind_m_plus] - 1j*c[flat_ind_m_minus])
        flat_ind += 1
    return complex_coeffs

def computeSymAxisLinCombSphHarm(complex_coeffs, A_surf, degree):
    from sympy.physics.quantum.spin import Rotation
    from scipy.optimize import fsolve
    complex_coeffs_norm = complex_coeffs/np.sqrt((np.sum([np.abs(coeff)**2 for coeff in complex_coeffs])/A_surf**2))
    alpha, beta, gamma = sp.symbols("alpha beta gamma")
    func = sp.lambdify([beta], sp.simplify(Rotation.D(degree, 0, 0, 0, beta, 0).expand(func=True)) - complex_coeffs_norm[degree])
    beta_0 = fsolve(func, np.pi/2)[0]
    print('beta', beta_0)
    func_2 = sp.lambdify([alpha], sp.conjugate(sp.simplify(Rotation.D(degree, 1, 0, alpha, beta_0, 0).expand(func=True))) - complex_coeffs_norm[degree + 1])
    func_3 = lambda alpha: np.real(func_2(alpha))
    alpha_0 = fsolve(func_3, np.pi/2)[0]
    print('alpha', alpha_0)
    func_5 = sp.lambdify([alpha], sp.conjugate(sp.simplify(Rotation.D(degree, 2, 0, alpha, beta_0, 0).expand(func=True))) - complex_coeffs_norm[degree + 2])
    func_6 = lambda alpha: np.real(func_5(alpha))
    alpha_1 = fsolve(func_6, np.pi/2)[0]
    print('alpha', alpha_1)
    gamma_0 = 0
    c_1 = np.cos(alpha_0)
    s_1 = np.sin(alpha_0)
    c_2 = np.cos(beta_0)
    s_2 = np.sin(beta_0)
    c_3 = np.cos(gamma_0)
    s_3 = np.sin(gamma_0)
    Z1_Y2_Z3 = np.array([[c_1*c_2*c_3 - s_1*s_3, -c_3*s_1 - c_1*c_2*s_3, c_1*s_2],
                        [c_1*s_3 + c_2*c_3*s_1, c_1*c_3 - c_2*s_1*s_3, s_1*s_2],
                        [-c_3*s_2, s_2*s_3, c_2]])
    sym_axis = Z1_Y2_Z3[:, 2]
    return sym_axis

def gaussQuad2D(func, range_i, range_j, n_points, n_digits=16):

    g_points, weights = np.array(gauss_legendre(n_points, n_digits), dtype=float)
    det = ((range_i[1] - range_i[0])/2)*((range_j[1] - range_j[0])/2)
    quad = 0
    for i_point, i_weight in zip(g_points, weights):
        for j_point, j_weight in zip(g_points, weights):
            quad += i_weight*j_weight*det*func(
                ((range_i[1] - range_i[0])/2)*i_point + ((range_i[1] + range_i[0])/2),
                ((range_j[1] - range_j[0])/2)*j_point + ((range_j[1] + range_j[0])/2))
    return quad


def printCoefficients(n, n_gauss=30):
    """Printing all the coefficients Mijk."""
    theta, phi = sp.symbols('theta phi')
    coeffs = "/home/zeluis/Documents/Tese/programa/src/M_i_j_k.dat"
    sph_harm_exprs = []
    counter = 0
    flat_ind = 0
    while True:
        degree = np.floor(np.sqrt(flat_ind))
        print(degree)
        if degree == 1:
            flat_ind += 1
            continue
        else:
            sph_harm_exprs.append(generateExprRealSphHarmonics(flat_ind))
            counter += 1
            flat_ind += 1
            if counter == n:
                break
    A = []
    B = []
    C = []
    for i_sph_harm in sph_harm_exprs:
        A.append(sp.simplify(diff_operator_A(i_sph_harm).expand(func=True)))
        B.append(sp.simplify(diff_operator_B(i_sph_harm).expand(func=True)))
        C.append(sp.simplify(diff_operator_C(i_sph_harm).expand(func=True)))

    with open(coeffs, "a") as dat:
        for i_ind, i_sph_harm in enumerate(sph_harm_exprs):
            for j_ind, j_sph_harm in enumerate(sph_harm_exprs):
                for k_ind, k_sph_harm in enumerate(sph_harm_exprs):
                    if i_ind <= j_ind and j_ind <= k_ind:
                        int_expr = (A[j_ind]*C[k_ind] - 2*B[j_ind]*B[k_ind] + A[k_ind]*C[j_ind])*sp.simplify(i_sph_harm.expand(func=True))*sp.sin(phi)
                        int_func = sp.lambdify([phi, theta], int_expr)
                        M_i_j_k = gaussQuad2D(int_func, [0., np.pi], [0., 2*np.pi], n_gauss)
                        print('M_i_j_k', np.round(M_i_j_k, decimals=10))
                        dat.write(str(np.round(M_i_j_k, decimals=10)) + '\n')

def computeMatrixA(c, ind_non0=2):
    A = []
    n = len(c)
    for k in range(n):
        if k == ind_non0:
            continue
        new_uvec = np.eye(n)[:, k] - c[k]/c[ind_non0]*np.eye(n)[:, ind_non0]
        norm_vec = new_uvec/np.linalg.norm(new_uvec)
        A.append(norm_vec)
    print(np.array(A).T)
    return np.array(A).T

def local_maxima(phi, theta, array):
    indices = ((array > np.roll(array,  1, 0)) &
            (array > np.roll(array, -1, 0)) &
            (array > np.roll(array,  1, 1)) &
            (array > np.roll(array, -1, 1)))
    print(array[indices])
    return [phi[indices], theta[indices], array[indices]]


def computeSurf():
    theta, phi = sp.symbols('theta phi')

    
    n = 46
    M = np.zeros((n, n, n))
    coeffs = "/home/zeluis/Documents/Tese/programa/src/M_i_j_k.dat"
    fin = open(coeffs, 'rt')

    lines = []
    for line in fin:
    	lines.append(line)

    counter = 0
    for i in range(n):
        for j in range(n):
            for k in range(n):
                if i <= j and j <= k:
                    val = float(lines[counter])
                    M[i, j, k] = val
                    M[k, i, j] = val
                    M[j, k, i] = val
                    M[i, k, j] = val
                    M[j, i, k] = val
                    M[k, j, i] = val
                    counter += 1

    
    fin.close()
    # Load
    print(M)
    c = np.array([2,
                 0, 0, 0, 0, 0,
                 0, 0, 1, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                    ])
    # Coefficients for the decomposition in spherical harmonics of the curvature

    if True:
        A = computeMatrixA(c, ind_non0=0)
        # Computing the matrix containing a basis for Z
        x = np.array([1/c[0],
                     0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                        ])
        # Initial guess
        disp = 100
        eps = 1e-8
        k_iter = 0
        max_iter = 100
        while True:
            g = np.zeros((n))
            H = np.zeros((n, n))
            V = 0
            for i in range(n):
                for j in range(n):
                    for k in range(n):
                        g[i] += 1/2*M[i, j, k]*x[j]*x[k]
                        # Computing the gradient vector
                        H[i, j] += M[i, j, k]*x[k]
                        # Computing the Hessian matrix
                        V += 1/6*M[i, j, k]*x[i]*x[j]*x[k]
                        # Computing the volume
            # print(H)
            # print(g)
            print('V', V)
            mat = A.T.dot((H - 2/3*1/V*g*g[:, np.newaxis]).dot(A))
            vec = -A.T.dot(g)
            d = np.linalg.solve(mat, vec)
            Ad = A.dot(d)
            VAd = 0
            for i in range(n):
                for j in range(n):
                    for k in range(n):
                        VAd += 1/6*M[i, j, k]*Ad[i]*Ad[j]*Ad[k]
                        # Computing the volume
            print('VAd', VAd, Ad)
            t = 0.1 #(-d.dot(A.T.dot(H.dot(A.dot(d)))) + np.sign(VAd)*np.sqrt((d.dot(A.T.dot(H.dot(A.dot(d)))))**2 - 12*d.dot(A.T.dot(g))*VAd))/(6*VAd)
            print('t', t)
            disp = t*A.dot(d)
            x += disp
            k_iter += 1
            print('disp', np.linalg.norm(disp))
            if np.linalg.norm(disp) < eps or k_iter > max_iter:
                break
    # else:
    #     x = np.array([1/c[0], 0, 0, 0, 0, 0])
    #     # Initial guess
    #     disp = 100
    #     eps = 1e-3
    #     k_iter = 0
    #     max_iter = 1000
    #     while True:
    #         g = np.zeros((n))
    #         for i in range(n):
    #             for j in range(n):
    #                 for k in range(n):
    #                     g[i] += 1/2*M[i, j, k]*x[j]*x[k]
    #                     # Computing the gradient vector
    #         print(g)
    #         f = g - c.dot(g)/c.dot(c)*c
    #         # Projection of the gradient on the plane (x-c)^Tc=0
    #         g_f = np.zeros((n))
    #         V_f = 0
    #         for i in range(n):
    #             for j in range(n):
    #                 for k in range(n):
    #                     g_f[i] += 1/2*M[i, j, k]*f[j]*f[k]
    #                     # Computing the gradient vector
    #                     V_f += 1/6*M[i, j, k]*f[j]*f[k]*f[i]
    #         W_f = x.dot(g_f)
    #         t = - (W_f + np.sqrt((W_f)**2 - 3*V_f*W_f))/(3*V_f)
    #         x += t*f
    #         disp = t*f
    #         if np.linalg.norm(disp) < eps or k_iter > max_iter:
    #             break
    print(x)
    theta, phi = sp.symbols('theta phi')
    sph = []
    flat_ind = 0
    counter = 0
    while True:
        degree = np.floor(np.sqrt(flat_ind))
        print(degree)
        if degree == 1:
            flat_ind += 1
            continue
        else:
            sph.append(generateExprRealSphHarmonics(flat_ind))
            counter += 1
            flat_ind += 1
            if counter == n:
                break
    support_expr = 0
    curvature_expr = 0
    for i_sph in range(n):
        support_expr += x[i_sph]*sph[i_sph]
        curvature_expr += c[i_sph]*sph[i_sph]
    

    diff_sup_func_phi = sp.lambdify([phi, theta], sp.diff(support_expr, phi))
    diff_sup_func_theta = sp.lambdify([phi, theta], sp.diff(support_expr, theta))
    sup_func = sp.lambdify([phi, theta], support_expr)
    curv_func = sp.lambdify([phi, theta], curvature_expr)

    # axis_degree = 3
    # complex_coeffs = realCoeffsToComplex(c[axis_degree**2 - 3:axis_degree**2 - 3 + 2*axis_degree + 1], axis_degree)
    # sym_axis = computeSymAxisLinCombSphHarm(complex_coeffs, c[0], axis_degree)
    


    import matplotlib.pyplot as plt
    import mpl_toolkits.mplot3d.axes3d as axes3d
    from scipy.special import sph_harm
    
    theta_vec, phi_vec = np.linspace(0, 2*np.pi, 1000), np.linspace(0, np.pi, 1000)
    theta, phi = np.meshgrid(theta_vec, phi_vec)
    R = np.zeros(theta.shape)
    for i, i_theta in enumerate(theta_vec):
        for j, j_phi in enumerate(phi_vec):
            R[i, j] = curv_func(theta[i,j], phi[i, j])

    # ind_max_curv = np.argmax(R)
    # theta_max = theta.flat[ind_max_curv]
    # phi_max = phi.flat[ind_max_curv]
    # print('theta', theta_max, 'phi', phi_max)
    X = theta
    Y = phi
    Z = R
    # X = R*np.cos(theta)*np.sin(phi)
    # Y = R*np.sin(phi)*np.sin(theta)
    # Z = R*np.cos(phi)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    plot = ax.plot_surface(
        X, Y, Z, rstride=1, cstride=1,
        linewidth=0, antialiased=False, alpha=0.5)
    
    phi_local_max, theta_local_max, local_maxima_curv = local_maxima(phi, theta, R)
    
    plt.show()
    # 
    # R = np.zeros(theta.shape)
    # for i, i_theta in enumerate(theta_vec):
    #     for j, j_phi in enumerate(phi_vec):
    #         R[i, j] = sup_func(j_phi, i_theta)
    # 
    # print('theta', theta_max, 'phi', phi_max)
    # X = theta
    # Y = phi
    # Z = R
    # # X = R*np.cos(theta)*np.sin(phi)
    # # Y = R*np.sin(phi)*np.sin(theta)
    # # Z = R*np.cos(phi)
    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1, projection='3d')
    # plot = ax.plot_surface(
    #     X, Y, Z, rstride=1, cstride=1,
    #     linewidth=0, antialiased=False, alpha=0.5)
    # 
    # plt.show()



    theta_vec, phi_vec = np.linspace(0, 2*np.pi, 100, endpoint=False), np.linspace(0+0.001, np.pi-0.001, 100)
    surf = []
    curv = []
    theta_used = []
    phi_used = []
    for theta in theta_vec:
        for phi in phi_vec:
            M = np.array(
                [[np.sin(phi)*np.cos(theta), np.sin(phi)*np.sin(theta), np.cos(phi)],
                 [np.cos(phi)*np.cos(theta), np.cos(phi)*np.sin(theta), -np.sin(phi)],
                 [-np.sin(theta), np.cos(theta), 0]])
            surf.append(M.T.dot([np.real(sup_func(phi, theta)), np.real(diff_sup_func_phi(phi, theta)), 1/np.sin(phi)*np.real(diff_sup_func_theta(phi, theta))]))
            curv.append(curv_func(phi, theta))
            theta_used.append(theta)
            phi_used.append(phi)

    surf = np.array(surf)
    curv = np.array(curv)
    ind_max = np.argmax(curv)
    print('curv', curv[ind_max])
    print('theta', theta_used[ind_max], 'phi', phi_used[ind_max])
    for theta_max, phi_max in zip(theta_local_max, phi_local_max):
        
        print('theta', theta_max, 'phi', phi_max)
        M = np.array(
            [[np.sin(phi_max)*np.cos(theta_max), np.sin(phi_max)*np.sin(theta_max), np.cos(phi_max)],
             [np.cos(phi_max)*np.cos(theta_max), np.cos(phi_max)*np.sin(theta_max), -np.sin(phi_max)],
             [-np.sin(theta_max), np.cos(theta_max), 0]])
        sym_axis = M.T.dot([np.real(sup_func(phi_max, theta_max)), np.real(diff_sup_func_phi(phi_max, theta_max)), 1/np.sin(phi_max)*np.real(diff_sup_func_theta(phi_max, theta_max))])

        print('axis', sym_axis)

    with open("/home/zeluis/Documents/Tese/programa/results/Y3m/vtk_file_com_20.vtk", "a") as msh_vtk:
        msh_vtk.write("# vtk DataFile Version 2.0")
        msh_vtk.write("\n3D triangulation data")
        msh_vtk.write("\nASCII")
        msh_vtk.write("\n\nDATASET POLYDATA")
        msh_vtk.write("\nPOINTS {0} {1}".format(len(surf), 'float'))
        for point in range(len(surf[:, 0])):
            msh_vtk.write("\n{0} {1} {2}".format(surf[point, 0], surf[point, 1], surf[point, 2]))
        msh_vtk.write("\n\nPOINT_DATA {0}".format(len(surf)))
        msh_vtk.write("\nSCALARS {0} {1} {2}".format('curvature', 'float', '1'))
        msh_vtk.write("\nLOOKUP_TABLE default")
        for point in range(len(surf[:, 0])):
            msh_vtk.write("\n{0}".format(curv[point]))


def func(x, y):
    return 1


if __name__ == '__main__':

    # psi_00, psi_2m1, psi_20 = sp.symbols('psi_00, psi_2m1 psi_20')
    # c = np.array([psi_00,
    #               0, psi_2m1, psi_20, 0, 0], dtype=object)
    # n = 4
    # theta, phi = sp.symbols('theta phi')
    # sph = []
    # flat_ind = 0
    # counter = 0
    # while True:
    #     degree = np.floor(np.sqrt(flat_ind))
    #     print(degree)
    #     if degree == 1:
    #         flat_ind += 1
    #         continue
    #     else:
    #         print('flat_ind', flat_ind)
    #         sph.append(generateExprRealSphHarmonics(flat_ind))
    #         counter += 1
    #         flat_ind += 1
    #         if counter == n:
    #             break
    # support_expr = 0
    # curvature_expr = 0
    # for i_sph in range(n):
    #     curvature_expr += c[i_sph]*sph[i_sph]
    # 
    # print(curvature_expr)
    # grad = gradSphericalCoords(curvature_expr)
    # 
    # print(grad)
    # print('grad0', grad[0])
    # print('grad1', grad[1])
    # print(sp.solve([grad[0], grad[1]], [phi, theta]))
    computeSurf()
