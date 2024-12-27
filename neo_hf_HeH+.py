import numpy as np
import math

"""
generate basis set for both hydrogens STO-3g
generate uncontracted basis set for one of the protons

calculate following integrals
    overlap integral of electronic and protonic bases
    kinetic energy of electrons and protons
    potential energy of electrons and protons
    two electron/proton integrals
    electron-proton integrals

create the overlap matrix for electronic and protonic bases
diagonalize the basis sets with X transform, different for proton and electron

calculate Hcore matrices for electronic and protonic basis sets

initialize coefficient and density matrices

calculate J and K matrices for electronic and protonic

calculate the fock matrix with Hcore J and K and interaction term

"""


class Primitive_gaussian:

    def __init__(self, alpha: float, R: np.ndarray, norm = None) -> None:
        self.alpha = alpha
        self.R = R
        if norm:
            self.norm = norm
        else:
            self.norm = (2*self.alpha/np.pi) ** 0.75


class Gaussian_basis:

    def __init__(self, R: np.ndarray, atom: str) -> None:
        function_coeffs = {"H": {"d": [0.1543289673E+00, 0.5353281423E+00, 0.4446345422E+00],
                                 "a": [0.3425250914E+01, 0.6239137298E+00, 0.1688554040E+00]},
                           "He": {"d": [0.1543289673E+00, 0.5353281423E+00, 0.4446345422E+00],
                                  "a": [0.6362421394E+01, 0.1158922999E+01, 0.3136497915E+00]},
                           "proton1": {"d": [1],
                                       "a": [8]},
                           "proton2": {"d": [1],
                                       "a": [4]},
                           }
        self.coeffs = function_coeffs[atom]["d"]
        self.gaussians = [Primitive_gaussian(
            i, R) for i in function_coeffs[atom]["a"]]


def gaussian_integral(g1: Primitive_gaussian, g2: Primitive_gaussian) -> float:
    RR = np.power(np.linalg.norm(g1.R-g2.R), 2)
    return g1.norm*g2.norm*(np.pi/(g1.alpha+g2.alpha))**1.5 * np.exp(-g1.alpha*g2.alpha/(g1.alpha+g2.alpha) * RR)


def overlap(basis_func1: Gaussian_basis, basis_func2: Gaussian_basis) -> float:
    overlap = 0.0
    for i in range(len(basis_func1.gaussians)):
        for j in range(len(basis_func2.gaussians)):
            gauss_overlap = gaussian_integral(
                basis_func1.gaussians[i], basis_func2.gaussians[j])
            overlap += gauss_overlap * \
                basis_func1.coeffs[i]*basis_func2.coeffs[j]
    return overlap


def create_overlap_matrix(basis_sets: list) -> np.ndarray:
    S = np.eye(len(basis_sets))
    for i in range(len(basis_sets)-1):
        for j in range(i+1, len(basis_sets)):
            S[i, j] = overlap(basis_sets[i], basis_sets[j])
            S[j, i] = S[i, j] # symmetric
    return S


def kinetic_energy_integral(gauss1: Primitive_gaussian, gauss2: Primitive_gaussian):
    RR = np.power(np.linalg.norm(gauss1.R-gauss2.R), 2)
    a = gauss1.alpha
    b = gauss2.alpha
    return gauss1.norm*gauss2.norm*a*b/(a+b)*(3-2*a*b/(a+b)*RR)*(np.pi/(a+b))**1.5*np.exp(-a*b/(a+b)*RR)


def kinetic_energy(basis_func1: Gaussian_basis, basis_func2: Gaussian_basis) -> float:
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
        for j in range(len(basis_sets)):
            T[i, j] = kinetic_energy(basis_sets[i], basis_sets[j])
    return T


def F0(t: float) -> float:
    if (t < 1e-6):
        return 1.0-t/3.0 # From Szabo Ostlund
    else:
        return 0.5*(np.pi/t)**0.5*math.erf(t**0.5)


def potential_energy_integral(gauss1: Primitive_gaussian, gauss2: Primitive_gaussian, n):
    RR = np.power(np.linalg.norm(gauss1.R-gauss2.R), 2)
    Rc = COORDS[n]
    Rp = (gauss1.alpha*gauss1.R+gauss2.alpha *
          gauss2.R)/(gauss1.alpha+gauss2.alpha)
    RpRc = np.power(np.linalg.norm(Rp-Rc), 2)
    Z_atom = CHARGES[n]
    a = gauss1.alpha
    b = gauss2.alpha
    return gauss1.norm*gauss2.norm*2*np.pi/(a+b)*Z_atom*np.exp(-a*b/(a+b)*RR)*F0((a+b)*RpRc)


def potential_energy(basis_func1: Gaussian_basis, basis_func2: Gaussian_basis, n: int) -> float:
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
    for i in range(len(basis_sets)):
        for j in range(len(basis_sets)):
            for index in I_CLASSICAL_NUCLEI:
                V_tot[i, j] += potential_energy(basis_sets[i],
                                                basis_sets[j], index)
    return V_tot


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


def calculate_two_electron_integral(f1: Gaussian_basis, f2: Gaussian_basis, f3: Gaussian_basis, f4: Gaussian_basis) -> float:
    '''
    This calculates two electron integrals for contracted gaussian basis sets (sto3g)
    '''
    rtn = 0.0
    g1, g2, g3, g4 = f1.gaussians, f2.gaussians, f3.gaussians, f4.gaussians
    l1, l2, l3, l4 = len(g1), len(g2), len(g3), len(g4)
    for i in range(l1):
        for j in range(l2):
            for ii in range(l3):
                for jj in range(l4):
                    gaussian_result = calculate_two_electron_integral_gaussian( # this function calculates the integral over two primitive gaussians
                        g1[i], g2[j], g3[ii], g4[jj])
                    rtn += gaussian_result * \
                        f1.coeffs[i]*f2.coeffs[j]*f3.coeffs[ii]*f4.coeffs[jj]
    return rtn


def create_two_integral_lookup(basis_sets1: list, basis_sets2: list) -> np.ndarray:
    l1 = len(basis_sets1)
    l2 = len(basis_sets2)
    two_integral_lookup = np.zeros([l1, l1, l2, l2])
    for i in range(l1):
        for j in range(l1):
            for ii in range(l2):
                for jj in range(l2):
                    two_integral_lookup[i, j, ii, jj] = calculate_two_electron_integral(
                        basis_sets1[i], basis_sets1[j], basis_sets2[ii], basis_sets2[jj])
    return two_integral_lookup


def calculate_J(P, integral_lookup: np.ndarray, basis_set):
    l = len(basis_set)
    J_tot = np.zeros([l, l])
    for i in range(l):
        for j in range(l):
            for ii in range(l):
                for jj in range(l):
                    J_tot[i, j] += P[ii, jj]*integral_lookup[i, j, ii, jj]
    return J_tot


def calculate_K(P, integral_lookup: np.ndarray, basis_set):
    l = len(basis_set)
    K_tot = np.zeros([l, l])
    for i in range(l):
        for j in range(l):
            for ii in range(l):
                for jj in range(l):
                    K_tot[i, j] += P[ii, jj]*integral_lookup[i, jj, ii, j]
    return K_tot


def calculate_mixed(P, integral_lookup: np.ndarray, basis_set):
    l = len(basis_set)
    mixed_tot = np.zeros([l, l])
    for i in range(l):
        for j in range(l):
            for ii in range(l):
                for jj in range(l):
                    mixed_tot[i, j] += P[ii, jj]*integral_lookup[i, j, ii, jj]
    return mixed_tot


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


def calculate_P(basis_sets, C: np.ndarray, electron: bool) -> np.ndarray:
    P = np.zeros([len(basis_sets), len(basis_sets)])
    if electron:
        for i in range(len(basis_sets)):
            for j in range(len(basis_sets)):
                for n in range(int(N_ELECTRONS/2)):
                    P[i, j] += C[i, n]*C[j, n]
        return P

    for i in range(len(basis_sets)):
        for j in range(len(basis_sets)):
            for n in range(len(I_QUANTIZED_NUCLEI)):
                P[i, j] += C[i, n]*C[j, n]
    return P


def calculate_E_neo(P_elec, F_elec, Hcore_elec, P_prot, F_prot, Hcore_prot) -> float:
    E = 0.0
    for i in range(N_ELECTRONS):
        for j in range(N_ELECTRONS):
            E += P_elec[i, j]*(Hcore_elec[i, j] + F_elec[i, j])
    for i in range(N_PROT_BASIS):
        for j in range(N_PROT_BASIS):
            E += 0.5*P_prot[i, j]*(Hcore_prot[i, j] + F_prot[i, j])
    return E


# proton at 0,0,0 is quantized
COORDS = np.array([[0.0, 0.0, 0.0],
                   [1.4, 0.0, 0.0]])

CHARGES = [1, 2]

I_QUANTIZED_NUCLEI = [0]
I_CLASSICAL_NUCLEI = [1]
N_ELECTRONS = 2
N_PROT_BASIS = 2

elec_basis = [Gaussian_basis(COORDS[0], "H"), Gaussian_basis(COORDS[1], "He")]
prot_basis = [Gaussian_basis(COORDS[0], 'proton1'),
              Gaussian_basis(COORDS[0], 'proton2')]

S_elec = create_overlap_matrix(elec_basis)
S_prot = create_overlap_matrix(prot_basis)

X_elec = calculate_X(S_elec)
X_prot = calculate_X(S_prot)

T_elec = calculate_T(elec_basis)
T_prot = calculate_T(prot_basis)

V_elec = calculate_V(elec_basis)
V_prot = calculate_V(prot_basis)

PROT_MASS = 1836.15267389

H_elec = T_elec - V_elec
H_prot = T_prot/PROT_MASS + V_prot

elec_integrals = create_two_integral_lookup(elec_basis, elec_basis)
prot_integrals = create_two_integral_lookup(prot_basis, prot_basis)

elec_prot_integrals = create_two_integral_lookup(elec_basis, prot_basis)
prot_elec_integrals = create_two_integral_lookup(prot_basis, elec_basis)

P_elec = np.ones((len(elec_basis), len(elec_basis)))
P_prot = np.ones((len(prot_basis), len(prot_basis)))
P_elec_old = P_elec
P_prot_old = P_prot

# INITIALIZE ELECTRON
J_elec = calculate_J(P_elec, elec_integrals, elec_basis)
K_elec = calculate_K(P_elec, elec_integrals, elec_basis)
mixed_elec = calculate_mixed(P_prot, elec_prot_integrals, elec_basis)

F_elec = H_elec + 2*J_elec - K_elec - mixed_elec
F_elec_tr = X_elec.T@F_elec@X_elec

e_elec, C_elec_tr = np.linalg.eigh(F_elec_tr)
C_elec = X_elec@C_elec_tr
P_elec = calculate_P(elec_basis, C_elec, True)

# INITIALIZE PROTON
J_prot = calculate_J(P_prot, prot_integrals, prot_basis)
K_prot = calculate_K(P_prot, prot_integrals, prot_basis)
mixed_prot = calculate_mixed(P_elec, prot_elec_integrals, prot_basis)

F_prot = H_prot + J_prot - K_prot - 2*mixed_prot
F_prot_tr = X_prot.T@F_prot@X_prot

e_prot, C_prot_tr = np.linalg.eigh(F_prot_tr)
C_prot = X_prot@C_prot_tr
P_prot = calculate_P(prot_basis, C_prot, False)

# CALCULATE FIRST ENERGY
E_old = calculate_E_neo(P_elec, F_elec, H_elec, P_prot, F_prot, H_prot)

norm_e = np.linalg.norm(P_elec - P_elec_old)
norm_p = np.linalg.norm(P_prot - P_prot_old)

niter = 10000
tol = 1E-10
tol_p = 1E-10
tol_e = 1E-10
E_diff = 1

while E_diff > tol:
    norm_e = 1
    while norm_e > tol_e:
        # ELECTRON
        P_elec_old = P_elec
        J_elec = calculate_J(P_elec, elec_integrals, elec_basis)
        K_elec = calculate_K(P_elec, elec_integrals, elec_basis)
        mixed_elec = calculate_mixed(P_prot, elec_prot_integrals, elec_basis)

        F_elec = H_elec + 2*J_elec - K_elec - mixed_elec
        F_elec_tr = X_elec.T@F_elec@X_elec

        e_elec, C_elec_tr = np.linalg.eigh(F_elec_tr)
        C_elec = X_elec@C_elec_tr
        P_elec = calculate_P(elec_basis, C_elec, True)

        norm_e = np.linalg.norm(P_elec - P_elec_old)
        E_new_el = calculate_E_neo(
            P_elec, F_elec, H_elec, P_prot, F_prot, H_prot)
        print(f"Energy for electronic iteration is {E_new_el}")

    norm_p = 1
    while norm_p > tol_p:
        # PROTON
        P_prot_old = P_prot
        J_prot = calculate_J(P_prot, prot_integrals, prot_basis)
        K_prot = calculate_K(P_prot, prot_integrals, prot_basis)
        mixed_prot = calculate_mixed(P_elec, prot_elec_integrals, prot_basis)

        F_prot = H_prot + J_prot - K_prot - 2*mixed_prot
        # F_prot = H_prot - 2*mixed_prot
        F_prot_tr = X_prot.T@F_prot@X_prot

        e_prot, C_prot_tr = np.linalg.eigh(F_prot_tr)
        C_prot = X_prot@C_prot_tr
        P_prot = calculate_P(prot_basis, C_prot, False)
        norm_p = np.linalg.norm(P_prot - P_prot_old)
        E_new_pr = calculate_E_neo(
            P_elec, F_elec, H_elec, P_prot, F_prot, H_prot)
        print(f"Energy for protonic iteration is {E_new_pr}")

    E_new = calculate_E_neo(P_elec=P_elec, F_elec=F_elec, Hcore_elec=H_elec,
                            P_prot=P_prot, F_prot=F_prot, Hcore_prot=H_prot)
    E_diff = abs(E_old-E_new)
    E_old = E_new
print(f"Energy converged to {E_new}")
