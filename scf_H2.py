import numpy as np
import math

# Hartree Fock of H2


class Primitive_gaussian:
    '''
    A class repersenting a primitive Gaussian (2*alpha/pi) ** 0.75 * exp(-alpha*|r-R|**2).

    Attributes:
        alpha : float 
            The exponent of the Gaussian.
        R : np.ndarray
            Coordinates for the middlepoint of the gaussian
        norm : float, default=(2*self.alpha/np.pi) ** 0.75
            Normalization constant of the gaussian
    '''

    def __init__(self, alpha: float, R: np.ndarray, norm: float = None) -> None:
        self.alpha = alpha
        self.R = R
        if norm:
            self.norm = norm
        else:
            self.norm = (2*self.alpha/np.pi) ** 0.75


class STO_3G:
    '''
    A class repersenting a slater type orbital with three primitive Gaussians.

    Attributes:
        R : np.ndarray
            Coordinates for the middlepoint of the orbital
    '''

    def __init__(self, R: np.ndarray, atom: str) -> None:
        function_coeffs = {"H": {"d": [0.1543289673E+00, 0.5353281423E+00, 0.4446345422E+00], "a": [0.3425250914E+01, 0.6239137298E+00, 0.1688554040E+00]}, "He": {
            "d": [0.1543289673E+00, 0.5353281423E+00, 0.4446345422E+00], "a": [0.6362421394E+01, 0.1158922999E+01, 0.3136497915E+00]}}
        d1, d2, d3 = function_coeffs[atom]["d"]
        a1, a2, a3 = function_coeffs[atom]["a"]
        self.coeffs = [d1, d2, d3]
        self.gaussians = [Primitive_gaussian(a1, R),
                          Primitive_gaussian(a2, R),
                          Primitive_gaussian(a3, R)]


def gaussian_integral(gauss1: Primitive_gaussian, gauss2: Primitive_gaussian) -> float:
    '''
    Calculates the overlap between two Gaussian curves.

    Parameters
    ----------
    gauss1 : Primitive_gaussian
        Primitive_gaussian object representing the first gaussian.
    gauss2 : Primitive_gaussian
        Primitive_gaussian object representing the second gaussian.

    Returns
    -------
    float
        The overlap of the two gaussians
    '''
    RR = np.power(np.linalg.norm(gauss1.R-gauss2.R), 2)
    return gauss1.norm*gauss2.norm*(np.pi/(gauss1.alpha+gauss2.alpha))**1.5 * np.exp(-gauss1.alpha*gauss2.alpha/(gauss1.alpha+gauss2.alpha) * RR)


def overlap(basis_func1: STO_3G, basis_func2: STO_3G) -> float:
    '''
    Calculates the overlap between two Gaussian basis functions with gausssian primitives.

    Parameters
    ----------
    basis_func1 : STO_3G
        STO_3G object representing the first basis function.
    basis_func2 : STO_3G
        STO_3G object representing the secomd basis function.

    Returns
    -------
    float
        The overlap of the two basis functions
    '''
    overlap = 0.0
    for i in range(len(basis_func1.gaussians)):
        for j in range(len(basis_func2.gaussians)):
            gauss_overlap = gaussian_integral(
                basis_func1.gaussians[i], basis_func2.gaussians[j])
            overlap += gauss_overlap * \
                basis_func1.coeffs[i]*basis_func2.coeffs[j]
    return overlap


def create_overlap_matrix(basis_sets: list) -> np.ndarray:
    '''
    Creates the overlap matrix between all the basis function pairs by taking symmetry into account i.e.
    <i|i> = 1
    <i|j> = <j|i>

    Parameters
    ----------
    basis_sets : list
        List of the basis sets in atomic order

    Returns
    -------
    np.ndarray
        Matrix of the overlaps
    '''
    S = np.eye(len(basis_sets))
    for i in range(len(basis_sets)-1):
        for j in range(i+1, len(basis_sets)):
            S[i, j] = overlap(basis_sets[i], basis_sets[j])
            S[j, i] = S[i, j]
    return S


def kinetic_energy_integral(gauss1: Primitive_gaussian, gauss2: Primitive_gaussian):
    '''
    Calculates the kinetic energy expectation value between two Gaussian curves.

    Parameters
    ----------
    gauss1 : Primitive_gaussian
        Primitive_gaussian object representing the first gaussian.
    gauss2 : Primitive_gaussian
        Primitive_gaussian object representing the second gaussian.

    Returns
    -------
    float
        Kinetic energy expectation value.
    '''
    RR = np.power(np.linalg.norm(gauss1.R-gauss2.R), 2)
    a = gauss1.alpha
    b = gauss2.alpha
    return gauss1.norm*gauss2.norm*a*b/(a+b)*(3-2*a*b/(a+b)*RR)*(np.pi/(a+b))**1.5*np.exp(-a*b/(a+b)*RR)


def kinetic_energy(basis_func1: STO_3G, basis_func2: STO_3G) -> float:
    '''
    Calculates the kinetic energy between two Gaussian basis functions with gausssian primitives.

    Parameters
    ----------
    basis_func1 : STO_3G
        STO_3G object representing the first basis function.
    basis_func2 : STO_3G
        STO_3G object representing the secomd basis function.

    Returns
    -------
    float
        The kinetic energy of the two basis functions
    '''
    Ek = 0.0
    for i in range(len(basis_func1.gaussians)):
        for j in range(len(basis_func2.gaussians)):
            gauss_Ek = kinetic_energy_integral(
                basis_func1.gaussians[i], basis_func2.gaussians[j])
            Ek += gauss_Ek * \
                basis_func1.coeffs[i]*basis_func2.coeffs[j]
    return Ek


def calculate_T(basis_sets: list) -> np.ndarray:
    '''
    Calculates kinetic energy T for the Hcore matrix. 

    Parameters
    ----------
    basis_sets : list
        List of the basis sets in atomic order

    Returns
    -------
    np.ndarray
        Matrix of the kinetic energies
    '''
    T = np.zeros([len(basis_sets), len(basis_sets)])
    for i in range(len(basis_sets)):
        for j in range(i, len(basis_sets)):
            T[i, j] = kinetic_energy(basis_sets[i], basis_sets[j])
            T[j, i] = T[i, j]
    return T


def F0(t: float) -> float:
    '''
    Error function like function from Szabos and Ostlunds book

    Parameters
    ----------
    t : float
        Value to be evaluated.

    Returns
    -------
    float
        Function value at t.
    '''
    if (t < 1e-6):
        return 1.0-t/3.0
    else:
        return 0.5*(np.pi/t)**0.5*math.erf(t**0.5)


def potential_energy_integral(gauss1: Primitive_gaussian, gauss2: Primitive_gaussian, n):
    '''
    Calculates the potential energy expectation value between two Gaussian curves.

    Parameters
    ----------
    gauss1 : Primitive_gaussian
        Primitive_gaussian object representing the first gaussian.
    gauss2 : Primitive_gaussian
        Primitive_gaussian object representing the second gaussian.

    Returns
    -------
    float
        Potential energy expectation value.
    '''
    RR = np.power(np.linalg.norm(gauss1.R-gauss2.R), 2)
    Rc = coords[n]
    Rp = (gauss1.alpha*gauss1.R+gauss2.alpha *
          gauss2.R)/(gauss1.alpha+gauss2.alpha)
    RpRc = np.power(np.linalg.norm(Rp-Rc), 2)
    Z_atom = Z[n]
    a = gauss1.alpha
    b = gauss2.alpha
    return gauss1.norm*gauss2.norm*-2*np.pi/(a+b)*Z_atom*np.exp(-a*b/(a+b)*RR)*F0((a+b)*RpRc)


def potential_energy(basis_func1: STO_3G, basis_func2: STO_3G, n: int) -> float:
    '''
    Calculates the potential energy between two Gaussian basis functions with gausssian primitives.

    Parameters
    ----------
    basis_func1 : STO_3G
        STO_3G object representing the first basis function.
    basis_func2 : STO_3G
        STO_3G object representing the secomd basis function.

    Returns
    -------
    float
        The potential energy of the two basis functions
    '''
    Ev = 0.0
    for i in range(len(basis_func1.gaussians)):
        for j in range(len(basis_func2.gaussians)):
            gauss_EV = potential_energy_integral(
                basis_func1.gaussians[i], basis_func2.gaussians[j], n)
            Ev += gauss_EV * \
                basis_func1.coeffs[i]*basis_func2.coeffs[j]
    return Ev


def calculate_V(basis_sets: list) -> np.ndarray:
    '''
    Calculates potential energy T for the Hcore matrix. 

    Parameters
    ----------
    basis_sets : list
        List of the basis sets in atomic order

    Returns
    -------
    np.ndarray
        Matrix of the potential energies
    '''
    V_tot = np.zeros([len(basis_sets), len(basis_sets)])
    for n in range(N):
        V = np.zeros([len(basis_sets), len(basis_sets)])
        for i in range(len(basis_sets)):
            for j in range(i, len(basis_sets)):
                V[i, j] = potential_energy(basis_sets[i], basis_sets[j], n)
                V[j, i] = V[i, j]
        V_tot += V
    return V_tot


def calculate_Hcore(basis_sets: list) -> np.ndarray:
    '''
    Calculates the Hcore matrix. 

    Parameters
    ----------
    basis_sets : list
        List of the basis sets in atomic order

    Returns
    -------
    np.ndarray
        Hcore matrix
    '''
    T = calculate_T(basis_sets)
    V = calculate_V(basis_sets)
    return T + V


def calculate_two_electron_integral_gaussian(g1: Primitive_gaussian, g2: Primitive_gaussian, g3: Primitive_gaussian, g4: Primitive_gaussian) -> float:
    '''
    Calculates the potential energy expectation value for a two electron pair of gaussian functions.

    Parameters
    ----------
    g1 : Primitive_gaussian
        Primitive_gaussian object representing the first gaussian.
    g2 : Primitive_gaussian
        Primitive_gaussian object representing the second gaussian.
    g3 : Primitive_gaussian
        Primitive_gaussian object representing the first gaussian.
    g4 : Primitive_gaussian
        Primitive_gaussian object representing the second gaussian.

    Returns
    -------
    float
        Energy for the two electron interaction.
    '''
    RaRb = np.power(np.linalg.norm(g1.R-g2.R), 2)
    RcRd = np.power(np.linalg.norm(g3.R-g4.R), 2)
    Rp = (g1.alpha*g1.R+g2.alpha *
          g2.R)/(g1.alpha+g2.alpha)
    Rq = (g3.alpha*g3.R+g4.alpha *
          g4.R)/(g3.alpha+g4.alpha)
    RpRq = np.power(np.linalg.norm(Rp-Rq), 2)
    norm = g1.norm*g2.norm*g3.norm*g4.norm
    a = g1.alpha
    b = g2.alpha
    c = g3.alpha
    d = g4.alpha
    return norm*2*np.pi**2.5/((a+b)*(c+d)*(a+b+c+d)**0.5)*np.exp(-a*b/(a+b)*RaRb-c*d/(c+d)*RcRd)*F0((a+b)*(c+d)/(a+b+c+d)*RpRq)


def calculate_two_electron_integral(f1: STO_3G, f2: STO_3G, f3: STO_3G, f4: STO_3G) -> float:
    '''
    Calculates the potential energy expectation value for a two electron pair represented with a basis set.

    Parameters
    ----------
    g1 : Primitive_gaussian
        Primitive_gaussian object representing the first gaussian.
    g2 : Primitive_gaussian
        Primitive_gaussian object representing the second gaussian.
    g3 : Primitive_gaussian
        Primitive_gaussian object representing the first gaussian.
    g4 : Primitive_gaussian
        Primitive_gaussian object representing the second gaussian.

    Returns
    -------
    float
        Energy for the two electron interaction.
    '''
    rtn = 0.0
    g1, g2, g3, g4 = f1.gaussians, f2.gaussians, f3.gaussians, f4.gaussians
    l1, l2, l3, l4 = len(g1), len(g2), len(g3), len(g4)
    for i in range(l1):
        for j in range(l2):
            for ii in range(l3):
                for jj in range(l4):
                    gaussian_result = calculate_two_electron_integral_gaussian(
                        g1[i], g2[j], g3[ii], g4[jj])
                    rtn += gaussian_result * \
                        f1.coeffs[i]*f2.coeffs[j]*f3.coeffs[ii]*f4.coeffs[jj]
    return rtn


def create_two_integral_lookup(basis_sets: list) -> np.ndarray:
    '''
    A function that creates an integral lookup table for all the two electron integrals (ab|cd), where a,b,c,d in {i,j}, in a highly inefficient way

    Parameters
    ----------
    basis_sets : list
        List of the basis sets in the same order as the atoms.

    Returns
    -------
    np.ndarray
        A lookup table with the indices [i,j,k,l] indicating the integral (ij|kl).    
    '''
    l = len(basis_sets)
    two_integral_lookup = np.zeros([l, l, l, l])
    for i in range(l):
        for j in range(l):
            for ii in range(l):
                for jj in range(l):
                    two_integral_lookup[i, j, ii, jj] = calculate_two_electron_integral(
                        basis_sets[i], basis_sets[j], basis_sets[ii], basis_sets[jj])
    return two_integral_lookup


def create_two_integral_lookup_symmetry(basis_sets: list) -> np.ndarray:
    '''
    A function that creates an integral lookup table for all the two electron integrals (ab|cd), where a,b,c,d in {i,j}, taking into account the symmetry of the operations
    (ji|ii) = (ij|ii) = (ii|ji) = (ii|ij)
    (jj|ii) = (ii|jj)
    (ji|ji) = (ji|ij) = (ij|ji) = (ij|ij)
    (jj|ji) = (jj|ij) = (ji|jj) = (ij|jj)

    Parameters
    ----------
    basis_sets : list
        List of the basis sets in the same order as the atoms.

    Returns
    -------
    np.ndarray
        A lookup table with the indices [a,b,c,d] indicating the integral (ab|cd).    
    '''
    l = len(basis_sets)
    two_integral_lookup = np.zeros([l, l, l, l])
    for i in range(l):
        two_integral_lookup[i, i, i, i] = calculate_two_electron_integral(
            basis_sets[i], basis_sets[i], basis_sets[i], basis_sets[i])
        for j in range(i+1, l):
            two_integral_lookup[j, i, i, i] = calculate_two_electron_integral(
                basis_sets[j], basis_sets[i], basis_sets[i], basis_sets[i])
            two_integral_lookup[i, j, i, i] = two_integral_lookup[i, i, j,
                                                                  i] = two_integral_lookup[i, i, i, j] = two_integral_lookup[j, i, i, i]

            two_integral_lookup[j, j, i, i] = calculate_two_electron_integral(
                basis_sets[j], basis_sets[j], basis_sets[i], basis_sets[i])
            two_integral_lookup[i, i, j, j] = two_integral_lookup[j, j, i, i]

            two_integral_lookup[j, i, j, i] = calculate_two_electron_integral(
                basis_sets[j], basis_sets[i], basis_sets[j], basis_sets[i])
            two_integral_lookup[j, i, i, j] = two_integral_lookup[i, j, j,
                                                                  i] = two_integral_lookup[i, j, i, j] = two_integral_lookup[j, i, j, i]

            two_integral_lookup[j, j, j, i] = calculate_two_electron_integral(
                basis_sets[j], basis_sets[j], basis_sets[j], basis_sets[i])
            two_integral_lookup[j, j, i, j] = two_integral_lookup[j, i, j,
                                                                  j] = two_integral_lookup[i, j, j, j] = two_integral_lookup[j, j, j, i]
    return two_integral_lookup


def calculate_F(basis_sets: list, Hcore: np.ndarray, P: np.ndarray) -> np.ndarray:
    '''
    Calculates the fock matrix 

    Parameters
    ----------
    basis_sets : list
        List of the basis sets in atomic order.
    Hcore : np.ndarray
        Hcore matrix.
    P : np.ndarray
        Density matrix.
    Returns
    -------
    np.ndarray
        Fock matrix
    '''
    G = np.zeros([len(basis_sets), len(basis_sets)])
    for i in range(len(basis_sets)):
        for j in range(len(basis_sets)):
            G_temp = 0.0
            for ii in range(len(basis_sets)):
                for jj in range(len(basis_sets)):
                    G_temp += P[ii, jj]*(integral_lookup[i, j, ii, jj] -
                                         0.5*integral_lookup[i, jj, ii, j])
            G[i, j] += G_temp

    return Hcore + G


def calculate_X(S: np.ndarray) -> np.ndarray:
    '''
    Diagonalizes the overlap matrix

    Parameters
    ----------
    s : np.ndarray
        Overlap matrix
    Returns
    -------
    np.ndarray
        Transformation matrix.
    '''
    s, U = np.linalg.eigh(S)
    sort_id = np.argsort(s)[::-1]
    s = s[sort_id]
    # U = U[sort_id, :]
    U = U[:, sort_id]
    s = s[np.where(s > 1.0E-4)]
    n = len(s)
    U = U[:, :n]
    X = U/s**0.5
    return X


def calculate_P(C: np.ndarray) -> np.ndarray:
    '''
    Calculates the density matrix

    Parameters
    ----------
    C : np.ndarray
        Coefficient matrix
    Returns
    -------
    np.ndarray
        Density matrix.
    '''
    P = np.zeros([len(basis_sets), len(basis_sets)])
    for n in range(int(N/2)):
        for i in range(len(basis_sets)):
            for j in range(len(basis_sets)):
                P[i, j] += 2*C[i, n]*C[j, n]
    return P


def calculate_E_0(P: np.ndarray, Hcore: np.ndarray, F: np.ndarray) -> float:
    '''
    Calculates E_0

    Parameters
    ----------
    P : np.ndarray
        Density matrix
    Hcore : np.ndarray
        Hcore matrix
    F : np.ndarray
        Fock matrix
    Returns
    -------
    float
        E_0
    '''
    E_0 = 0.0
    for i in range(N):
        for j in range(N):
            E_0 += P[i, j]*(Hcore[i, j] + F[i, j])
    return 0.5*E_0


def calculate_Etot(E_0: float) -> float:
    '''
    Calculates total energy of the molecule

    Parameters
    ----------
    E_0 : float
        E_0
    Returns
    -------
    float
        Total energy 
    '''
    E_rep = 0
    for i in range(Natoms):
        for j in range(i+1, Natoms):
            E_rep += Z[i]*Z[j]/float(np.linalg.norm(coords[i]-coords[j]))
    return E_0 + E_rep


# xyz coordinates of the atoms
coords = np.array([[0.0, 0.0, 0.0],
                   [1.4, 0.0, 0.0]])

# nucelar charge of the atoms
Z = [1, 1]

# number of atoms
Natoms = 2

# number of electrons
N = 2

# define basis set for each electron
basis_sets = [STO_3G(coords[0], "H"), STO_3G(coords[1], "H")]
# calculate overlap matrix for each basis function pair
S = create_overlap_matrix(basis_sets)
# calculate tranformation matrix X
X = calculate_X(S)
# initialize density matrix P to 0, meaning first iteration is done with only core potential
P = np.zeros([len(basis_sets), len(basis_sets)])

# initialize a two integral lookup for all nbasis^4 quadruples of two integrals
integral_lookup = create_two_integral_lookup_symmetry(basis_sets)

# calculate Hcore
Hcore = calculate_Hcore(basis_sets)
# calculate initial fock matrix
F = calculate_F(basis_sets=basis_sets, Hcore=Hcore, P=P)
# transform fock matrix to fit the equation F'C' = C'e
F_transformed = X.T@F@X
# calculate energy eigenvalues and C matrix
e, C_transformed = np.linalg.eigh(F)
# transform C' back to C
C = X@C_transformed
# calculate energies for first iteration
E_0 = calculate_E_0(P=P, Hcore=Hcore, F=F)
Etot_old = calculate_Etot(E_0)

# calculate new density matrix using the calculated C
P = calculate_P(C)

niter = 25
tol = 1E-6

for i in range(niter):
    # calculate new fock matrix with updatet P
    F = calculate_F(basis_sets=basis_sets, Hcore=Hcore, P=P)
    # transform fock matrix to fit the equation F'C' = C'e
    F_transformed = X.T@F@X
    # calculate energy eigenvalues and C matrix
    e, C_transformed = np.linalg.eigh(F_transformed)
    # transform C' back to C
    C = X@C_transformed

    # calculate new energies
    E_0 = calculate_E_0(P=P, Hcore=Hcore, F=F)
    Etot_new = calculate_Etot(E_0)
    print()

    print(f"Energy for iteration {i} is {Etot_new}")
    # compare old energy to new energy
    if abs(Etot_new-Etot_old) < tol:
        print(f"Energy converged to {Etot_new}")
        break
    Etot_old = Etot_new

    # calculate new density matrix using the calculated C
    P = calculate_P(C)
