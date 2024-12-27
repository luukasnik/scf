import numpy as np
import math

# NEO Hartree Fock of H2


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


class Gaussian_basis:
    '''
    A class repersenting a slater type orbital with three primitive Gaussians.

    Attributes:
        R : np.ndarray
            Coordinates for the middlepoint of the orbital
    '''

    def __init__(self, R: np.ndarray, atom: str) -> None:
        function_coeffs = {"H": {"d": [0.1543289673E+00, 0.5353281423E+00, 0.4446345422E+00],
                                 "a": [0.3425250914E+01, 0.6239137298E+00, 0.1688554040E+00]},
                           "He": {"d": [0.1543289673E+00, 0.5353281423E+00, 0.4446345422E+00],
                                  "a": [0.6362421394E+01, 0.1158922999E+01, 0.3136497915E+00]},
                           "proton": {"d": [1, 1],
                                      "a": [8, 4]}
                           }
        self.coeffs = function_coeffs[atom]["d"]
        self.gaussians = [Primitive_gaussian(
            i, R) for i in function_coeffs[atom]["a"]]


class Uncontracted_Gaussian_basis:
    '''
    A class repersenting a slater type orbital with three primitive Gaussians.

    Attributes:
        R : np.ndarray
            Coordinates for the middlepoint of the orbital
    '''

    def __init__(self, R: np.ndarray, atom: str) -> None:
        function_coeffs = {"H": {"d": [0.1543289673E+00, 0.5353281423E+00, 0.4446345422E+00],
                                 "a": [0.3425250914E+01, 0.6239137298E+00, 0.1688554040E+00]},
                           "He": {"d": [0.1543289673E+00, 0.5353281423E+00, 0.4446345422E+00],
                                  "a": [0.6362421394E+01, 0.1158922999E+01, 0.3136497915E+00]},
                           "proton1": {"d": [1],
                                       "a": [4]},
                           "proton2": {"d": [1],
                                       "a": [8]}
                           }
        self.coeffs = function_coeffs[atom]["d"]
        self.gaussians = [Primitive_gaussian(
            i, R) for i in function_coeffs[atom]["a"]]


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


def overlap(basis_func1: Gaussian_basis, basis_func2: Gaussian_basis) -> float:
    '''
    Calculates the overlap between two Gaussian basis functions with gausssian primitives.

    Parameters
    ----------
    basis_func1 : Gaussian_basis
        Gaussian_basis object representing the first basis function.
    basis_func2 : Gaussian_basis
        Gaussian_basis object representing the secomd basis function.

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


def kinetic_energy(basis_func1: Gaussian_basis, basis_func2: Gaussian_basis) -> float:
    '''
    Calculates the kinetic energy between two Gaussian basis functions with gausssian primitives.

    Parameters
    ----------
    basis_func1 : Gaussian_basis
        Gaussian_basis object representing the first basis function.
    basis_func2 : Gaussian_basis
        Gaussian_basis object representing the secomd basis function.

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


def potential_energy(basis_func1: Gaussian_basis, basis_func2: Gaussian_basis, n: int) -> float:
    '''
    Calculates the potential energy between two Gaussian basis functions with gausssian primitives.

    Parameters
    ----------
    basis_func1 : Gaussian_basis
        Gaussian_basis object representing the first basis function.
    basis_func2 : Gaussian_basis
        Gaussian_basis object representing the secomd basis function.

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
    for n in range(N_elec):
        V = np.zeros([len(basis_sets), len(basis_sets)])
        for i in range(len(basis_sets)):
            for j in range(i, len(basis_sets)):
                V[i, j] = -potential_energy(basis_sets[i], basis_sets[j], n)
                V[j, i] = V[i, j]
        V_tot += V
    return V_tot


def calculate_V_elec(basis_sets: list) -> np.ndarray:
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
    for n in [0]:
        V = np.zeros([len(basis_sets), len(basis_sets)])
        for i in range(len(basis_sets)):
            for j in range(i, len(basis_sets)):
                V[i, j] = potential_energy(basis_sets[i], basis_sets[j], n)
                V[j, i] = V[i, j]
        V_tot += V
    return V_tot


def calculate_Hcore_elec(basis_sets: list) -> np.ndarray:
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
    V = calculate_V_elec(basis_sets)
    return T + V


def calculate_Hcore_prot(basis_sets: list) -> np.ndarray:
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
    return T/1836.15267389 + V


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


def calculate_two_electron_integral(f1: Gaussian_basis, f2: Gaussian_basis, f3: Gaussian_basis, f4: Gaussian_basis) -> float:
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


def create_two_integral_lookup_mixed(basis_elec, basis_prot):
    l_el = len(basis_elec)
    l_pr = len(basis_prot)
    # l = l_el if l_el > l_pr else l_pr
    two_integral_lookup = np.zeros([l_el, l_el, l_pr, l_pr])
    for i in range(l_el):
        for j in range(l_el):
            for ii in range(l_pr):
                for jj in range(l_pr):
                    two_integral_lookup[i, j, ii, jj] = calculate_two_electron_integral(
                        basis_elec[i], basis_elec[j], basis_prot[ii], basis_prot[jj])
    return two_integral_lookup


def create_two_integral_lookup_protonic(prot_basis):
    l = len(prot_basis)
    two_integral_lookup = np.zeros([l, l, l, l])
    for i in range(l):
        for j in range(l):
            for ii in range(l):
                for jj in range(l):
                    two_integral_lookup[i, j, ii, jj] = calculate_two_electron_integral(
                        prot_basis[i], prot_basis[j], prot_basis[ii], prot_basis[jj])
    return two_integral_lookup


def calculate_F_elec(basis_sets: list, prot_basis: list, Hcore: np.ndarray, P_elec: np.ndarray, P_prot: np.ndarray) -> np.ndarray:
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
                    G_temp += P_elec[ii, jj]*(integral_lookup_elec[i, j, ii, jj] -
                                              0.5*integral_lookup_elec[i, jj, ii, j])
            G[i, j] += G_temp

    G_mixed = np.zeros([len(basis_sets), len(basis_sets)])
    for i in range(len(basis_sets)):
        for j in range(len(basis_sets)):
            G_temp = 0.0
            for ii in range(len(prot_basis)):
                for jj in range(len(prot_basis)):
                    G_temp += P_prot[ii, jj] * \
                        integral_lookup_mixed[i, j, ii, jj]

            G_mixed[i, j] += G_temp

    return Hcore + G - G_mixed


def calculate_F_prot(prot_basis: list, basis_sets: list, Hcore_prot: np.ndarray, P_elec: np.ndarray, P_prot: np.ndarray) -> np.ndarray:
    '''
    Calculates the fock matrix 

    Parameters
    ----------
    prot_basis : list
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
    G = np.zeros([len(prot_basis), len(prot_basis)])
    for i in range(len(prot_basis)):
        for j in range(len(prot_basis)):
            G_temp = 0.0
            for ii in range(len(prot_basis)):
                for jj in range(len(prot_basis)):
                    G_temp += P_prot[ii, jj]*(integral_lookup_prot[i, j, ii, jj] -
                                              integral_lookup_prot[i, jj, ii, j])
            G[i, j] += G_temp

    G_mixed = np.zeros([len(prot_basis), len(prot_basis)])
    for i in range(len(prot_basis)):
        for j in range(len(prot_basis)):
            G_temp = 0.0
            for ii in range(len(basis_sets)):
                for jj in range(len(basis_sets)):
                    G_temp += P_elec[ii, jj] * \
                        integral_lookup_mixed[i, j, ii, jj]

            G_mixed[i, j] += G_temp

    return Hcore_prot + G - G_mixed


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


def calculate_P(basis_sets, C: np.ndarray) -> np.ndarray:
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
    for n in range(int(N_elec/2)):
        for i in range(len(basis_sets)):
            for j in range(len(basis_sets)):
                P[i, j] += 2*C[i, n]*C[j, n]
    return P


def calculate_P_prot(prot_basis, C: np.ndarray) -> np.ndarray:
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
    P = np.zeros([len(prot_basis), len(prot_basis)])
    for n in range(int(N_Qprot)):
        for i in range(len(prot_basis)):
            for j in range(len(prot_basis)):
                P[i, j] += C[i, n]*C[j, n]
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
    for i in range(N_elec):
        for j in range(N_elec):
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


def calculate_E_neo(P_elec, F_elec, Hcore_elec, P_prot, F_prot, Hcore_prot) -> float:
    E = 0.0
    for i in range(N_elec):
        for j in range(N_elec):
            E += 0.5*P_elec[i, j]*(Hcore_elec[i, j] + F_elec[i, j])
    for i in range(Natoms):
        for j in range(Natoms):
            E += 0.5*P_prot[i, j]*(Hcore_prot[i, j] + F_prot[i, j])
    return E


# xyz coordinates of the atoms
coords = np.array([[0.0, 0.0, 0.0],
                   [1.4, 0.0, 0.0]])

# nucelar charge of the atoms
Z = [1, 1]

# number of atoms
Natoms = 2

# number of electrons
N_elec = 2
N_Catoms = 1
N_Qprot = 1

# define basis set for each electron
basis_sets = [Gaussian_basis(coords[0], "H"), Gaussian_basis(coords[1], "H")]
prot_basis = [Uncontracted_Gaussian_basis(coords[0], 'proton1'),
              Uncontracted_Gaussian_basis(coords[0], 'proton2')]
# calculate overlap matrix for each basis function pair
S_elec = create_overlap_matrix(basis_sets)
S_prot = create_overlap_matrix(prot_basis)
# calculate tranformation matrix X
X_elec = calculate_X(S_elec)
X_prot = calculate_X(S_prot)
# initialize density matrix P to 0, meaning first iteration is done with only core potential
P_elec = np.zeros([len(basis_sets), len(basis_sets)])
P_prot = np.zeros([len(prot_basis), len(prot_basis)])

# initialize a two integral lookup for all nbasis^4 quadruples of two integrals
integral_lookup_elec = create_two_integral_lookup_symmetry(basis_sets)
integral_lookup_mixed = create_two_integral_lookup_mixed(
    basis_sets, prot_basis)
integral_lookup_prot = create_two_integral_lookup_protonic(prot_basis)

# calculate Hcore
Hcore_elec = calculate_Hcore_elec(basis_sets)
Hcore_prot = calculate_Hcore_prot(prot_basis)

# calculate initial fock matrix
F_elec = calculate_F_elec(basis_sets=basis_sets, prot_basis=prot_basis, Hcore=Hcore_elec,
                          P_elec=P_elec, P_prot=P_prot)
F_prot = calculate_F_prot(prot_basis, basis_sets, Hcore_prot, P_elec, P_prot)
# transform fock matrix to fit the equation F'C' = C'e
F_elec_transformed = X_elec.T@F_elec@X_elec
F_prot_transformed = X_prot.T@F_prot@X_prot
# calculate energy eigenvalues and C matrix
e_elec, C_elec_transformed = np.linalg.eigh(F_elec)
# transform C' back to C
C_elec = X_elec@C_elec_transformed
# calculate energies for first iteration
E_0 = calculate_E_0(P=P_elec, Hcore=Hcore_elec, F=F_elec)
Etot_old = calculate_Etot(E_0)

# calculate new density matrix using the calculated C
P_elec = calculate_P(basis_sets=basis_sets, C=C_elec)

niter = 25
tol = 1E-6

# First converge non NEO calculation
for i in range(niter):
    # calculate new fock matrix with updatet P
    F_elec = calculate_F_elec(basis_sets=basis_sets, prot_basis=prot_basis, Hcore=Hcore_elec,
                              P_elec=P_elec, P_prot=P_prot)
    # transform fock matrix to fit the equation F'C' = C'e
    F_elec_transformed = X_elec.T@F_elec@X_elec
    # calculate energy eigenvalues and C matrix
    e_elec, C_elec_transformed = np.linalg.eigh(F_elec_transformed)
    # transform C' back to C
    C_elec = X_elec@C_elec_transformed

    # calculate new energies
    E_0 = calculate_E_0(P=P_elec, Hcore=Hcore_elec, F=F_elec)
    Etot_new = calculate_Etot(E_0)

    print(f"Energy for iteration {i} is {Etot_new}")
    # compare old energy to new energy
    if abs(Etot_new-Etot_old) < tol:
        print(f"Energy converged to {Etot_new}")
        break
    Etot_old = Etot_new

    # calculate new density matrix using the calculated C
    P_elec = calculate_P(basis_sets, C_elec)


niter = 25
tol = 1E-6
for i in range(niter):
    # PROTON
    F_prot = calculate_F_prot(prot_basis=prot_basis, basis_sets=basis_sets,
                              Hcore_prot=Hcore_prot, P_elec=P_elec, P_prot=P_prot)
    # transform fock matrix to fit the equation F'C' = C'e
    F_prot_transformed = X_prot.T@F_prot@X_prot
    # calculate energy eigenvalues and C matrix
    e_prot, C_prot_transformed = np.linalg.eigh(F_prot_transformed)
    # transform C' back to C
    C_prot = X_prot@C_prot_transformed
    # calculate new density matrix using the calculated C
    P_prot = calculate_P_prot(prot_basis, C_prot)

    # ELECTRON

    # calculate new fock matrix with updatet P
    F_elec = calculate_F_elec(basis_sets=basis_sets, prot_basis=prot_basis, Hcore=Hcore_elec,
                              P_elec=P_elec, P_prot=P_prot)
    # transform fock matrix to fit the equation F'C' = C'e
    F_elec_transformed = X_elec.T@F_elec@X_elec
    # calculate energy eigenvalues and C matrix
    e_elec, C_elec_transformed = np.linalg.eigh(F_elec_transformed)
    # transform C' back to C
    C_elec = X_elec@C_elec_transformed
    # calculate new density matrix using the calculated C
    P_elec = calculate_P(basis_sets, C_elec)

    # calculate new energies
    Etot_new = calculate_E_neo(
        P_elec, F_elec, Hcore_elec, P_prot, F_prot, Hcore_prot)

    print(f"Energy for iteration {i} is {Etot_new}")
    # compare old energy to new energy
    if abs(Etot_new-Etot_old) < tol:
        print(f"Energy converged to {Etot_new}")
        break
    Etot_old = Etot_new
