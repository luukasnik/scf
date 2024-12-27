import numpy as np
import math

# Hartree Fock of HeH+


class Primitive_gaussian:

    def __init__(self, alpha: float, R: np.ndarray, r: float = None, norm: float = None) -> None:
        self.alpha = alpha
        self.R = R
        self.r = r
        if norm:
            self.norm = norm
        else:
            self.norm = (2*self.alpha/np.pi) ** 0.75


class STO_3G:

    def __init__(self, R: np.ndarray, atom: str) -> None:
        function_coeffs = {"H": {"d": [0.1543289673E+00, 0.5353281423E+00, 0.4446345422E+00], "a": [0.3425250914E+01, 0.6239137298E+00, 0.1688554040E+00]}, "He": {
            "d": [0.1543289673E+00, 0.5353281423E+00, 0.4446345422E+00], "a": [0.6362421394E+01, 0.1158922999E+01, 0.3136497915E+00]}}
        d1, d2, d3 = function_coeffs[atom]["d"]
        a1, a2, a3 = function_coeffs[atom]["a"]
        self.coeffs = [d1, d2, d3]
        self.gaussians = [Primitive_gaussian(a1, R),
                          Primitive_gaussian(a2, R),
                          Primitive_gaussian(a3, R)]


def primitive_gaussian(alpha, r, R):
    return (2*alpha/np.pi) ** (3/4)*np.exp(-alpha*np.linalg.norm(r-R))


def multiply_gaussian(gauss1, gauss2):
    norm = gauss1.norm*gauss2.norm
    K = np.exp(-gauss1.alpha*gauss2.alpha/(gauss1.alpha +
               gauss2.alpha)*np.linalg.norm(gauss1.R-gauss2.R))
    R = (gauss1.alpha*gauss1.R+gauss2.alpha *
         gauss2.R)/(gauss1.alpha+gauss2.alpha)
    alpha = gauss1.alpha+gauss2.alpha
    return Primitive_gaussian(alpha, R, norm=norm*K)


def gaussian_integral(gauss1: Primitive_gaussian, gauss2: Primitive_gaussian) -> float:
    RR = np.power(np.linalg.norm(gauss1.R-gauss2.R), 2)
    # print(RR)
    return gauss1.norm*gauss2.norm*(np.pi/(gauss1.alpha+gauss2.alpha))**1.5 * np.exp(-gauss1.alpha*gauss2.alpha/(gauss1.alpha+gauss2.alpha) * RR)


def overlap(basis_func1: STO_3G, basis_func2: STO_3G) -> float:
    overlap = 0.0
    for i in range(len(basis_func1.gaussians)):
        for j in range(len(basis_func2.gaussians)):
            gauss_overlap = gaussian_integral(
                basis_func1.gaussians[i], basis_func2.gaussians[j])
            overlap += gauss_overlap * \
                basis_func1.coeffs[i]*basis_func2.coeffs[j]
    return overlap


def overlap_matrix(basis_sets: list) -> np.ndarray:
    S = np.eye(len(basis_sets))
    for i in range(len(basis_sets)-1):
        for j in range(i+1, len(basis_sets)):
            S[i, j] = overlap(basis_sets[i], basis_sets[j])
            S[j, i] = S[i, j]
    return S


def kinetic_energy_integral(gauss1: Primitive_gaussian, gauss2: Primitive_gaussian):
    RR = np.power(np.linalg.norm(gauss1.R-gauss2.R), 2)
    a = gauss1.alpha
    b = gauss2.alpha
    return gauss1.norm*gauss2.norm*a*b/(a+b)*(3-2*a*b/(a+b)*RR)*(np.pi/(a+b))**1.5*np.exp(-a*b/(a+b)*RR)


def kinetic_energy(basis_func1: STO_3G, basis_func2: STO_3G) -> float:
    Ek = 0.0
    for i in range(len(basis_func1.gaussians)):
        for j in range(len(basis_func2.gaussians)):
            gauss_Ek = kinetic_energy_integral(
                basis_func1.gaussians[i], basis_func2.gaussians[j])
            Ek += gauss_Ek * \
                basis_func1.coeffs[i]*basis_func2.coeffs[j]
    return Ek


def calculate_T(basis_sets: list) -> np.ndarray:
    T = np.zeros([len(basis_sets), len(basis_sets)])
    for i in range(len(basis_sets)):
        for j in range(i, len(basis_sets)):
            T[i, j] = kinetic_energy(basis_sets[i], basis_sets[j])
            T[j, i] = T[i, j]
    return T


def F0(t):
    if (t < 1e-6):
        return 1.0-t/3.0
    else:
        return 0.5*(np.pi/t)**0.5*math.erf(t**0.5)


def potential_energy_integral(gauss1: Primitive_gaussian, gauss2: Primitive_gaussian, n):
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
    Ev = 0.0
    for i in range(len(basis_func1.gaussians)):
        for j in range(len(basis_func2.gaussians)):
            gauss_EV = potential_energy_integral(
                basis_func1.gaussians[i], basis_func2.gaussians[j], n)
            Ev += gauss_EV * \
                basis_func1.coeffs[i]*basis_func2.coeffs[j]
    return Ev


def calculate_V(basis_sets: list) -> np.ndarray:
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
    T = calculate_T(basis_sets)
    V = calculate_V(basis_sets)
    return T + V


def calculate_two_electron_integral_gaussian(g1: Primitive_gaussian, g2: Primitive_gaussian, g3: Primitive_gaussian, g4: Primitive_gaussian) -> float:
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
    l = len(basis_sets)
    two_integral_lookup = np.zeros([l, l, l, l])
    for n in range(int(N/2)):
        for i in range(l):
            for j in range(l):
                for ii in range(l):
                    for jj in range(l):
                        # Symmtery has not been taken into account here
                        two_integral_lookup[i, j, ii, jj] = calculate_two_electron_integral(
                            basis_sets[i], basis_sets[j], basis_sets[ii], basis_sets[jj])
    return two_integral_lookup


def calculate_F(basis_sets: list, Hcore: np.ndarray, P: np.ndarray) -> np.ndarray:
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


def calculate_P(C):
    P = np.zeros([len(basis_sets), len(basis_sets)])
    for n in range(int(N)):
        for i in range(len(basis_sets)):
            for j in range(len(basis_sets)):
                P[i, j] += C[i, n]*C[j, n]
    return P


def calculate_E_0(P, Hcore, F):
    E_0 = 0.0
    for i in range(N):
        for j in range(N):
            E_0 += P[i, j]*(Hcore[i, j] + F[i, j])
    return 0.5*E_0


def calculate_Etot(E_0):
    E_rep = 0
    for i in range(Natoms):
        for j in range(i+1, Natoms):
            E_rep += Z[i]*Z[j]/float(np.linalg.norm(coords[i]-coords[j]))
    return E_0 + E_rep


# xyz coordinates of the atoms
coords = np.array([[0.0, 0.0, 0.0]])

# nucelar charge of the atoms
Z = [1]

# number of atoms
Natoms = 1

# number of electrons
N = 1

# define basis set for each electron
basis_sets = [STO_3G(coords[0], "H")]

# calculate overlap matrix for each basis function pair
S = overlap_matrix(basis_sets)
# calculate tranformation matrix X
X = calculate_X(S)
# initialize density matrix P to 0, meaning first iteration is done with only core potential
P = np.ones([len(basis_sets), len(basis_sets)])

# initialize a two integral lookup for all nbasis^4 quadruples of two integrals
integral_lookup = create_two_integral_lookup(basis_sets)

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
