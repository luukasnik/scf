#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 16:09:02 2020

@author: fabijanpavosevic
"""
import psi4
import psi4 as psi4_total
import psi4 as psi4_e
import psi4 as psi4_p
import numpy as np

import os

# S_e_file = os.path.join(
#     '/Users/fabijanpavosevic/software/qc-code/libint-2.4.2/tests', 'S_e.dat')
# S_e_data = np.genfromtxt(fname=S_e_file, delimiter=',', dtype='unicode')
# S_e_data = S_e_data.astype(np.float)

# X_e_file = os.path.join(
#     '/Users/fabijanpavosevic/software/qc-code/libint-2.4.2/tests', 'X_e.dat')
# X_e_data = np.genfromtxt(fname=X_e_file, delimiter=',', dtype='unicode')
# X_e_data = X_e_data.astype(np.float)

# H_e_file = os.path.join(
#     '/Users/fabijanpavosevic/software/qc-code/libint-2.4.2/tests', 'H_e.dat')
# H_e_data = np.genfromtxt(fname=H_e_file, delimiter=',', dtype='unicode')
# H_e_data = H_e_data.astype(np.float)

# g_ee_file = os.path.join(
#     '/Users/fabijanpavosevic/software/qc-code/libint-2.4.2/tests', 'g_ee.dat')
# g_ee_data = np.genfromtxt(fname=g_ee_file, delimiter=',', dtype='unicode')
# g_ee_data = g_ee_data.astype(np.float)

# V_p_file = os.path.join(
#     '/Users/fabijanpavosevic/software/qc-code/libint-2.4.2/tests', 'V_p.dat')
# V_p_data = np.genfromtxt(fname=V_p_file, delimiter=',', dtype='unicode')
# V_p_data = V_p_data.astype(np.float)

# S_p_file = os.path.join(
#     '/Users/fabijanpavosevic/software/qc-code/libint-2.4.2/tests', 'S_p.dat')
# S_p_data = np.genfromtxt(fname=S_p_file, delimiter=',', dtype='unicode')
# S_p_data = S_p_data.astype(np.float)

# H_p_file = os.path.join(
#     '/Users/fabijanpavosevic/software/qc-code/libint-2.4.2/tests', 'H_p.dat')
# H_p_data = np.genfromtxt(fname=H_p_file, delimiter=',', dtype='unicode')
# H_p_data = H_p_data.astype(np.float)

# g_ep_file = os.path.join(
#     '/Users/fabijanpavosevic/software/qc-code/libint-2.4.2/tests', 'g_ep.dat')
# g_ep_data = np.genfromtxt(fname=g_ep_file, delimiter=',', dtype='unicode')
# g_ep_data = g_ep_data.astype(np.float)

# ==> Set Basic Psi4 Options <==
# Memory specification
psi4_total.set_memory(int(5e8))
numpy_memory = 2

# Set output file
psi4.core.set_output_file('output.dat', False)

# defining geometry of the whole molecule
mol_total = psi4_total.geometry("""
0 1
H       0.0   0.00000   0.0       
H       0.7   0.00000  0.0   

no_reorient
no_com
""")

# Set computation options
psi4_total.set_options({'basis': 'STO-3G',
                        'scf_type': 'pk',
                        'e_convergence': 1e-8})

wfn_total = psi4_total.core.Wavefunction.build(
    mol_total, psi4_total.core.get_global_option('basis'))

E_HF_total = psi4_total.energy('SCF')
print(E_HF_total)


# defining geometry of the classical centers
mol_e = psi4_e.geometry("""
-1 1   
H       0.0   0.00000   0.0      
@H      0.7   0.00000  0.0       

no_reorient
no_com
""")

# Set computation options
psi4_e.set_options({'basis': 'STO-3G',
                    'scf_type': 'pk',
                    'e_convergence': 1e-8})

wfn_e = psi4_e.core.Wavefunction.build(
    mol_e, psi4_e.core.get_global_option('basis'))


mol_p = psi4_p.geometry("""
-1 1

H    0.7   0.00000  0.0      


no_reorient
no_com
""")

Chrgfield = psi4_p.QMMMbohr()
Chrgfield.extern.addCharge(8, 0.00000, -0.07579,   0.00000)
Chrgfield.extern.addCharge(1, 0.86681, 0.60144,    0.00000)
psi4_p.core.set_global_option_python('EXTERN', Chrgfield.extern)


def basisspec_psi4(mol, role):
    basstrings = {}
    mol.set_basis_by_label("H", "dz", role=role)
    basstrings['dz'] = """
cartesian
****
****
H     0
S   1   1.00
      4.0000           1.0000000
P   1   1.00
      4.0000           1.0000000
****
"""
    return basstrings


psi4_p.qcdb.libmintsbasisset.basishorde['SP'] = basisspec_psi4
psi4_p.set_options({'basis': 'SP',
                    'scf_type': 'pk',
                    'print': 7,
                    'e_convergence': 1e-8})

wfn_p = psi4_p.core.Wavefunction.build(
    mol_p, psi4_p.core.get_global_option('basis'))

E_HF_neo_c = psi4_p.energy('SCF')
print('NEO center SCF energy')
print(E_HF_neo_c)

# defining electronic and protonic basis sets
basis_e = wfn_e.basisset()
# defining electronic 1 and 2-body integrals
mints_e = psi4_e.core.MintsHelper(basis_e)
S_e = np.asarray(mints_e.ao_overlap())
nbf_e = S_e.shape[0]
# S_e = np.reshape(S_e_data, (nbf_e, nbf_e))

T_e = np.asarray(mints_e.ao_kinetic())
V_e = np.asarray(mints_e.ao_potential())
H_e = T_e + V_e
# H_e = np.reshape(H_e_data, (nbf_e, nbf_e))

I_ee = np.asarray(mints_e.ao_eri())
# g_ee = np.reshape(g_ee_data, (nbf_e, nbf_e, nbf_e, nbf_e))
# nbf_e = S_e.shape[0]
nocc_e = wfn_e.nalpha()

# defining electronic and protonic basis sets
basis_p = wfn_p.basisset()

# defining protonic 1 and 2-body integrals
mints_p = psi4_p.core.MintsHelper(basis_p)
S_p = np.asarray(mints_p.ao_overlap())
# nbf_p = S_p_ints.shape[0]
# S_p = np.reshape(S_p_data, (nbf_p, nbf_p))

nbf_p = S_p.shape[0]
nocc_p = wfn_p.nalpha()
T_p = np.asarray(mints_p.ao_kinetic())
V_p = np.asarray(mints_p.ao_potential())
print(V_p)
# V_p = np.reshape(V_p_data, (nbf_p, nbf_p))
print('''[[0.71428571 0.65389433]
 [0.65389433 0.7142857 ]]''')


H_p = T_p/1836.15267389 - V_p
# H_p = np.reshape(H_p_data, (nbf_p, nbf_p))
I_pp = np.asarray(mints_p.ao_eri())


I_ep = np.asarray(mints_e.ao_eri(basis_e, basis_e, basis_p, basis_p))
I_pe = np.asarray(mints_p.ao_eri(basis_e, basis_e, basis_p, basis_p))

# g_ep = np.reshape(g_ep_data, (nbf_e, nbf_e, nbf_p, nbf_p))
g_ep = np.array([[[[-1.1596647,  -1.04448545],
                   [-1.04448545, -1.10732956]],

                  [[-0.58157256, -0.52801479],
                   [-0.52801479, -0.56777022]]],


                 [[[-0.58157256, -0.52801479],
                   [-0.52801479, -0.56777022]],

                  [[-0.6499709,  -0.59381771],
                     [-0.59381771, -0.6460019]]]])

wfn_p_pot = psi4_p.core.Wavefunction.build(
    mol_p, psi4_p.core.get_global_option('basis'))

print('Number of electronic basis functions: %3d' % (nbf_e))
print('Number of protonic basis functions: %5d' % (nbf_p))

print('Number of occupied electronic orbitals: %1d' % (nocc_e))
print('Number of occupied protonic orbitals: %3d' % (nocc_p))

# Orthogonalization matrix
X_e = mints_e.ao_overlap()
X_e.power(-0.5, 1.e-16)
X_e = np.asarray(X_e)
# X_e = np.reshape(X_e_data, (nbf_e, nbf_e))

X_p = mints_p.ao_overlap()
X_p.power(-0.5, 1.e-16)
X_p = np.asarray(X_p)


# nuclear-neculear repulsion energy
E_nuc = mol_e.nuclear_repulsion_energy()
print(E_nuc)

E_e, C_e = np.linalg.eigh(X_e.dot(H_e).dot(X_e))
C_e = X_e.dot(C_e)
Cocc_e = C_e[:, :nocc_e]
# D_e_old = Cocc_e.dot(Cocc_e.transpose())
D_e_old = np.zeros((nbf_e, nbf_e))

E_p, C_p = np.linalg.eigh(X_p.dot(H_p).dot(X_p))
C_p = X_p.dot(C_p)
Cocc_p = C_p[:, :nocc_p]
# D_p_old = Cocc_p.dot(Cocc_p.transpose())
D_p_old = np.zeros((nbf_p, nbf_p))

diff_SCF_E = 1
E_old = 0
E_threshold = 1e-10
MAXITER = 50
e_threshold = 1e-08
p_threshold = 1e-08
count = 0
vpp = False

while diff_SCF_E > E_threshold:

    count = 0
    norm_e = 1
    while norm_e > e_threshold:
        J_e = np.einsum('pqrs,rs->pq', I_ee, D_e_old, optimize=True)
        K_e = np.einsum('prqs,rs->pq', I_ee, D_e_old, optimize=True)
        J_ep = np.einsum('pqrs,rs->pq', g_ep, D_p_old, optimize=True)

        F_e = H_e + 2*J_e - K_e - J_ep
        # print(X_e.transpose().dot(F_e).dot(X_e))

        E_e, Ct_e = np.linalg.eigh(X_e.transpose().dot(F_e).dot(X_e))
        C_e = X_e.dot(Ct_e)
        Cocc_e = C_e[:, :nocc_e]
        D_e = Cocc_e.dot(Cocc_e.transpose())

        norm_e = np.linalg.norm(D_e - D_e_old)
        D_e_old = D_e
        E_new = np.einsum('pq,pq->', (H_e + F_e), D_e, optimize=True)
        E_new += np.einsum('pq,pq->', 0.5*(H_p + F_p), D_p, optimize=True)
        E_new += E_nuc
        print(E_new)

    print('done with electronic')
    norm_p = 1
    while norm_p > p_threshold:
        J_p = np.einsum('pqrs,rs->pq', I_pp, D_p_old, optimize=True)
        K_p = np.einsum('prqs,rs->pq', I_pp, D_p_old, optimize=True)
        J_ep = np.einsum('pqrs,pq->rs', g_ep, D_e_old, optimize=True)

        if vpp == False:
            F_p = H_p - 2*J_ep
        elif vpp == True:
            F_p = H_p + J_p - K_p - 2*J_ep

        E_p, Ct_p = np.linalg.eigh(X_p.dot(F_p).dot(X_p))
        C_p = X_p.dot(Ct_p)
        Cocc_p = C_p[:, :nocc_p]
        D_p = Cocc_p.dot(Cocc_p.transpose())

        norm_p = np.linalg.norm(D_p - D_p_old)
        D_p_old = D_p
        E_new = np.einsum('pq,pq->', (H_e + F_e), D_e, optimize=True)
        E_new += np.einsum('pq,pq->', 0.5*(H_p + F_p), D_p, optimize=True)
        E_new += E_nuc
        print(E_new)

    print('done with protonic')

    E_new = np.einsum('pq,pq->', (H_e + F_e), D_e, optimize=True)
    E_new += np.einsum('pq,pq->', 0.5*(H_p + F_p), D_p, optimize=True)
    E_new += E_nuc
    diff_SCF_E = abs(E_new - E_old)
    E_old = E_new
    print(E_new)
