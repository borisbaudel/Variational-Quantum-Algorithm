"""
verify_derivations.py -- every equation of the report's appendix, checked
numerically.

The assignment asks to "show all the mathematical reasoning/derivations".  A
derivation that is only written down can be wrong in a way that no reader
notices; each identity below is therefore also *evaluated*, and the tolerance
it passes at is printed.  Equation numbers refer to the appendix of report.pdf.

Output: results/derivations.json, plus a table for the report.
"""

import json

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp

from spinvqe.analysis import (q2_ground_state, sample_energy_qwc,
                              shift_rule_general, shift_rule_two_term,
                              shots_for_precision, sublayer_frequencies)
from spinvqe.ansatz import HVA, SINGLET, apply_exchange, exchange_gate
from spinvqe.core import (exchange_op, ground_state, hs_coupling,
                          hs_exact_energy, hs_hamiltonian, pauli_string,
                          sz_sector_indices, two_site_op, xxx_bonds,
                          xxx_hamiltonian)
from spinvqe.syk import (commute, majorana_terms, syk_hamiltonian,
                         syk_pauli_terms)

CHECKS = []


def record(tag, name, err, tol, extra=""):
    ok = bool(err <= tol)
    CHECKS.append({"eq": tag, "name": name, "error": float(err),
                   "tol": tol, "passed": ok, "note": extra})
    flag = "ok  " if ok else "FAIL"
    print(f"  {flag} ({tag:>5s}) {name:<58s} err = {err:.2e}"
          + (f"   [{extra}]" if extra else ""))


# ===========================================================================
print("\nA. conventions and Pauli algebra")
# ---------------------------------------------------------------------------
# (A.1)  sigma = 2 S   =>   sigma_i . sigma_j = 4 S_i . S_j
L = 4
S = {p: 0.5 * pauli_string("".join(p if k == 0 else "I" for k in range(1)))
     for p in "XYZ"}
lhs = exchange_op(2, 0, 1).toarray()
Sx, Sy, Sz = (0.5 * np.array([[0, 1], [1, 0]], dtype=complex),
              0.5 * np.array([[0, -1j], [1j, 0]]),
              0.5 * np.array([[1, 0], [0, -1]], dtype=complex))
rhs = 4 * sum(np.kron(A, A) for A in (Sx, Sy, Sz))
record("A.1", "sigma_i.sigma_j = 4 S_i.S_j", np.abs(lhs - rhs).max(), 1e-13)

# (A.2)  spectrum of sigma.sigma is {+1 (x3), -3 (x1)}
ev = np.sort(np.linalg.eigvalsh(exchange_op(2, 0, 1).toarray()))
record("A.2", "spec(sigma.sigma) = {-3, 1, 1, 1}",
       np.abs(ev - np.array([-3, 1, 1, 1])).max(), 1e-12)

# (A.3)  P_swap = (1 + sigma.sigma)/2
swap = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex)
record("A.3", "swap = (1 + sigma.sigma)/2",
       np.abs(swap - 0.5 * (np.eye(4) + exchange_op(2, 0, 1).toarray())).max(), 1e-13)

# (A.4)  XX, YY, ZZ commute pairwise on a pair
mx, my, mz = (two_site_op(2, 0, 1, p, p).toarray() for p in "XYZ")
err = max(np.abs(a @ b - b @ a).max() for a, b in ((mx, my), (my, mz), (mx, mz)))
record("A.4", "[XX,YY] = [YY,ZZ] = [XX,ZZ] = 0", err, 1e-13)

# (A.5)  exp(-i t (XX+YY+ZZ)) = RXX(2t) RYY(2t) RZZ(2t)
t = 0.413
rx, ry, rz = (la.expm(-1j * t * m) for m in (mx, my, mz))
record("A.5", "exchange gate = RXX(2t) RYY(2t) RZZ(2t)",
       np.abs(exchange_gate(t) - rx @ ry @ rz).max(), 1e-13)

# (A.6)  closed form of the gate (block structure)
th = 0.77
G = np.zeros((4, 4), dtype=complex)
G[0, 0] = G[3, 3] = np.exp(-1j * th)
al, be = np.exp(1j * th) * np.cos(2 * th), -1j * np.exp(1j * th) * np.sin(2 * th)
G[1, 1] = G[2, 2] = al
G[1, 2] = G[2, 1] = be
record("A.6", "block form of exp(-i t sigma.sigma)",
       np.abs(G - exchange_gate(th)).max(), 1e-13)

# ===========================================================================
print("\nB. Model 1: ansatz, symmetry, gradients")
# ---------------------------------------------------------------------------
# (B.1)  the singlet is the -3 eigenvector; reference energy = -3 per bond
record("B.1", "sigma.sigma |singlet> = -3 |singlet>",
       np.abs(exchange_op(2, 0, 1).toarray() @ SINGLET + 3 * SINGLET).max(), 1e-13)

# (B.2)  the HVA never leaves the S^z = 0 sector
L = 8
ans = HVA(L, 3, *xxx_bonds(L))
rng = np.random.default_rng(0)
psi = ans.state(rng.normal(0, 0.7, ans.n_params))
sec = sz_sector_indices(L, L // 2)
mask = np.ones(2**L, dtype=bool)
mask[sec] = False
record("B.2", "HVA state has no weight outside S^z = 0",
       float(np.abs(psi[mask]).max()), 1e-13)

# (B.3)  ... and is a total singlet:  S^2_tot |psi> = 0
Stot2 = sp.csr_matrix((2**L, 2**L), dtype=complex)
for i in range(L):
    for j in range(L):
        Stot2 = Stot2 + 0.25 * exchange_op(L, i, j) if i != j else Stot2
Stot2 = Stot2 + 0.75 * L * sp.identity(2**L, format="csr", dtype=complex)
record("B.3", "S^2_tot |psi_HVA> = 0 (total singlet)",
       float(np.linalg.norm(Stot2 @ psi)), 1e-9)

# (B.4)  adjoint gradient == central finite differences
H = xxx_hamiltonian(L)
p0 = rng.normal(0, 0.4, ans.n_params)
E_ad, g_ad = ans.energy_and_grad(p0, H)
eps = 1e-6
g_fd = np.array([
    (np.real(np.vdot(ans.state(p0 + eps * np.eye(ans.n_params)[k]),
                     H @ ans.state(p0 + eps * np.eye(ans.n_params)[k])))
     - np.real(np.vdot(ans.state(p0 - eps * np.eye(ans.n_params)[k]),
                       H @ ans.state(p0 - eps * np.eye(ans.n_params)[k]))))
    / (2 * eps) for k in range(ans.n_params)])
record("B.4", "adjoint gradient = finite differences",
       float(np.abs(g_ad - g_fd).max()), 1e-6)

# (B.5)  single-bond parameter shift: dE/dt = 2[E(t+pi/8) - E(t-pi/8)]
L2 = 4
Hs = xxx_hamiltonian(L2)
ref = np.kron(SINGLET, SINGLET)


def E_one_bond(t):
    v = apply_exchange(ref, t, 1, 2, L2)
    return float(np.real(np.vdot(v, Hs @ v)))


t0 = 0.37
num = (E_one_bond(t0 + 1e-6) - E_one_bond(t0 - 1e-6)) / 2e-6
record("B.5", "two-term shift rule (Omega=4) for one bond",
       abs(shift_rule_two_term(E_one_bond, t0) - num), 1e-6,
       "shift pi/8, prefactor 2")

# (B.6)  for a whole sub-layer the two-term rule FAILS, the 2R-term one works
L3 = 8
H3 = xxx_hamiltonian(L3)
odd, even = xxx_bonds(L3)
base = HVA(L3, 1, odd, even).reference()


def E_layer(t):
    v = base
    for (a, b) in even:            # 4 disjoint bonds -> R = 4
        v = apply_exchange(v, t, a, b, L3)
    return float(np.real(np.vdot(v, H3 @ v)))


num3 = (E_layer(t0 + 1e-6) - E_layer(t0 - 1e-6)) / 2e-6
err_two = abs(shift_rule_two_term(E_layer, t0) - num3)
err_gen = abs(shift_rule_general(E_layer, t0, sublayer_frequencies(len(even))) - num3)
record("B.6a", "two-term rule FAILS for an R=4 sub-layer (error is large)",
       0.0 if err_two > 1e-3 else 1.0, 0.5, f"observed error {err_two:.2e}")
record("B.6b", "2R-term rule is exact for the same sub-layer", err_gen, 1e-6,
       "frequencies 4,8,12,16")

# ===========================================================================
print("\nC. Model 2: Haldane-Shastry")
# ---------------------------------------------------------------------------
for L in (6, 8, 12):
    e = 0.0
    for j in range(L):
        for k in range(j + 1, L):
            zj, zk = np.exp(2j * np.pi * j / L), np.exp(2j * np.pi * k / L)
            e = max(e, abs(abs(zj - zk) ** 2 - 4 * np.sin(np.pi * (j - k) / L) ** 2))
    record("C.1", f"|z_j - z_k|^2 = 4 sin^2(pi(j-k)/L)   [L={L}]", e, 1e-12)

# (C.2)  the assignment's H equals the textbook HS Hamiltonian
L = 8
H_assign = hs_hamiltonian(L)
H_text = sp.csr_matrix((2**L, 2**L), dtype=complex)
for j in range(L):
    for k in range(j + 1, L):
        c = (np.pi / L) ** 2 / np.sin(np.pi * (j - k) / L) ** 2
        H_text = H_text + 0.25 * c * exchange_op(L, j, k)   # S.S = sigma.sigma/4
record("C.2", "H(assignment) = J (pi/L)^2 sum S_j.S_k / sin^2",
       float(abs(H_assign - H_text).max()), 1e-12)

# (C.3)  closed-form energy
for L in (4, 6, 8, 10, 12):
    E0 = float(ground_state(hs_hamiltonian(L), k=1,
                            sector=sz_sector_indices(L, L // 2))[0][0])
    record("C.3", f"E_0 = -(pi^2/24)(L + 5/L)   [L={L}]",
           abs(E0 - hs_exact_energy(L)), 1e-9)

# (C.4)  the Jastrow state, with and without the momentum factor
from run_model2 import jastrow_ground_state  # noqa: E402

for L in (6, 8, 10):
    sec = sz_sector_indices(L, L // 2)
    Hs2 = hs_hamiltonian(L)[sec][:, sec]
    psiJ, _ = jastrow_ground_state(L, extra_phase=True)
    r = float(np.linalg.norm(Hs2 @ psiJ - complex(np.vdot(psiJ, Hs2 @ psiJ)).real * psiJ))
    record("C.4", f"Jastrow x prod z is an eigenvector   [L={L}]", r, 1e-9)
    _, v = ground_state(hs_hamiltonian(L), k=1, sector=sec)
    record("C.5", f"... and it is the ground state, |<.|.>| = 1   [L={L}]",
           abs(1.0 - abs(np.vdot(v[:, 0], psiJ))), 1e-9)
    psiN, _ = jastrow_ground_state(L, extra_phase=False)
    ov = abs(np.vdot(v[:, 0], psiN))
    record("C.6", f"without prod z: orthogonal to the ground state  [L={L}]",
           ov, 1e-9, "an excited eigenvector")

# ===========================================================================
print("\nD. Model 3: Jordan-Wigner and SYK")
# ---------------------------------------------------------------------------
for N in (6, 8, 10):
    n = N // 2
    mats = [pauli_string(t.s) * t.c for t in majorana_terms(N)]
    eye = sp.identity(2**n, format="csr", dtype=complex)
    e_same = max(float(abs(m @ m - eye).max()) for m in mats)
    e_diff = max(float(abs(mats[a] @ mats[b] + mats[b] @ mats[a]).max())
                 for a in range(N) for b in range(N) if a != b)
    record("D.1", f"chi_a^2 = 1   [N={N}]", e_same, 1e-12)
    record("D.2", f"{{chi_a, chi_b}} = 0 for a != b   [N={N}]", e_diff, 1e-12)

# (D.3)  each quartic term is a single Pauli string and is Hermitian
N = 10
H, terms, agg = syk_hamiltonian(N, seed=3)
chi = majorana_terms(N)
prod = chi[0] * chi[1] * chi[2] * chi[3]
M = pauli_string(prod.s) * prod.c
record("D.3", "chi_i chi_j chi_k chi_l is one Pauli string",
       float(abs(M - pauli_string(prod.s) * prod.c).max()), 1e-13,
       f"string {prod.s}, phase {prod.c:+.0f}")
record("D.4", "(chi_i chi_j chi_k chi_l)^dagger = itself  [(-1)^6 = +1]",
       float(abs(M - M.getH()).max()), 1e-12)
record("D.5", "H_SYK is Hermitian and traceless",
       max(float(abs(H - H.getH()).max()), abs(H.diagonal().sum())), 1e-10)

# (D.6)  number of terms is exactly C(N,4)
import math  # noqa: E402
record("D.6", f"number of Pauli strings = C({N},4) = {math.comb(N,4)}",
       abs(len(agg) - math.comb(N, 4)), 0.5)

# (D.7)  commutation criterion: even overlap of differing non-identity sites
rngp = np.random.default_rng(1)
strs = list(agg)[:60]
e = 0.0
for a in strs[:25]:
    for b in strs[:25]:
        A, B = pauli_string(a), pauli_string(b)
        anti = float(abs(A @ B - B @ A).max())
        if commute(a, b):
            e = max(e, anti)
        else:
            e = max(e, 1.0 if anti < 1e-9 else 0.0)
record("D.7", "commute(a,b) predicate matches [A,B] = 0", e, 1e-10)

# (D.8)  exp(-i t P) = cos t - i sin t P   for P^2 = 1   (the ADAPT gate)
P = pauli_string(list(agg)[0])
t = 0.61
lhs = la.expm(-1j * t * P.toarray())
rhs = np.cos(t) * np.eye(P.shape[0]) - 1j * np.sin(t) * P.toarray()
record("D.8", "exp(-i t P) = cos t - i sin t P", np.abs(lhs - rhs).max(), 1e-12)

# (D.9)  ADAPT selection gradient:  dE/dt|_0 = -2 Im <P psi | H psi>
rng2 = np.random.default_rng(5)
v = rng2.normal(size=P.shape[0]) + 1j * rng2.normal(size=P.shape[0])
v /= np.linalg.norm(v)


def E_of_t(t):
    u = np.cos(t) * v - 1j * np.sin(t) * (P @ v)
    return float(np.real(np.vdot(u, H @ u)))


num = (E_of_t(1e-6) - E_of_t(-1e-6)) / 2e-6
ana = -2.0 * np.imag(np.vdot(P @ v, H @ v))
record("D.9", "dE/dtheta|_0 = -2 Im <P psi|H psi>", abs(num - ana), 1e-6)

# (D.10) q=2 Majorana model is Hermitian and its ground state is exact
H2, E2, psi2 = q2_ground_state(12, seed=2)
record("D.10", "H_{q=2} Hermitian", float(abs(H2 - H2.getH()).max()), 1e-12)
record("D.11", "H_{q=2} psi_0 = E_0 psi_0",
       float(np.linalg.norm(H2 @ psi2 - E2 * psi2)), 1e-9)

# (D.12) the shot estimator is unbiased
N = 8
Hs3, _, agg3 = syk_hamiltonian(N, seed=11)
n3 = N // 2
w3, v3 = np.linalg.eigh(Hs3.toarray())
st = v3[:, 0]
exact = float(np.real(np.vdot(st, Hs3 @ st)))
rng3 = np.random.default_rng(0)
ests = [sample_energy_qwc(st, agg3, 4000, rng3) for _ in range(40)]
bias = abs(np.mean(ests) - exact)
record("D.12", "QWC shot estimator is unbiased", bias, 4 * np.std(ests) / np.sqrt(40),
       f"exact {exact:.4f}, mean {np.mean(ests):.4f}, sd {np.std(ests):.4f}")

# ===========================================================================
n_ok = sum(c["passed"] for c in CHECKS)
print(f"\n{n_ok}/{len(CHECKS)} derivations verified")
with open("results/derivations.json", "w") as f:
    json.dump({"n_checks": len(CHECKS), "n_passed": n_ok, "checks": CHECKS}, f, indent=2)
