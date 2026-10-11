"""
test_core.py -- the checks that guard every claim in the report.

Run with:  python -m pytest tests -q       (or simply: python tests/test_core.py)

Each test corresponds to a statement made in the report, so that nothing there
rests on a number that was typed rather than computed.
"""

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp

from spinvqe.ansatz import (HVA, SINGLET, apply_2q, apply_exchange,
                            exchange_gate)
from spinvqe.core import (exchange_op, ground_state, hs_exact_energy,
                          hs_hamiltonian, pauli_string, sz_sector_indices,
                          xxx_bonds, xxx_hamiltonian)
from spinvqe.syk import (check_clifford_algebra, commute, majorana_terms,
                         pauli_string_triplets, qubit_wise_commute,
                         syk_hamiltonian)
from spinvqe.vqe import central_grad, make_energy


# --------------------------------------------------------------- primitives
def test_hermiticity():
    for L in (4, 6, 8):
        for H in (xxx_hamiltonian(L), hs_hamiltonian(L)):
            assert abs(H - H.getH()).max() < 1e-12


def test_exchange_gate_is_unitary():
    for th in (0.0, 0.31, -1.7, np.pi):
        G = exchange_gate(th)
        assert np.abs(G.conj().T @ G - np.eye(4)).max() < 1e-13


def test_exchange_gate_matches_expm():
    L, a, b, th = 4, 1, 3, 0.37
    rng = np.random.default_rng(1)
    psi = rng.normal(size=2**L) + 1j * rng.normal(size=2**L)
    psi /= np.linalg.norm(psi)
    ref = la.expm(-1j * th * exchange_op(L, a, b).toarray()) @ psi
    assert np.abs(apply_2q(psi, exchange_gate(th), a, b, L) - ref).max() < 1e-12


def test_fast_exchange_matches_general():
    """The specialised block-structured gate equals the generic contraction,
    including for the periodic bond where a > b."""
    L = 10
    rng = np.random.default_rng(7)
    psi = rng.normal(size=2**L) + 1j * rng.normal(size=2**L)
    psi /= np.linalg.norm(psi)
    for (a, b) in [(2, 3), (0, 7), (9, 0), (4, 5)]:
        r1 = apply_2q(psi, exchange_gate(0.41), a, b, L)
        r2 = apply_exchange(psi, 0.41, a, b, L)
        assert np.abs(r1 - r2).max() < 1e-12


# --------------------------------------------------------------- model 1
def test_xxx_L4_exact():
    """L=4 PBC ring: E0 = -8 in the Pauli normalisation (= -2 in spin units)."""
    w, _ = ground_state(xxx_hamiltonian(4), k=1)
    assert abs(w[0] + 8.0) < 1e-10


def test_sector_matches_full_space():
    for L in (4, 6, 8):
        H = xxx_hamiltonian(L)
        full, _ = ground_state(H, k=1)
        sec, _ = ground_state(H, k=1, sector=sz_sector_indices(L, L // 2))
        assert abs(full[0] - sec[0]) < 1e-10


def test_hva_reference_is_singlet_product():
    L = 6
    ans = HVA(L, 1, *xxx_bonds(L))
    psi = ans.reference()
    assert abs(np.linalg.norm(psi) - 1) < 1e-12
    # energy of the odd layer should be -3 per bond
    Hodd = sum(exchange_op(L, a, b) for (a, b) in ans.odd)
    assert abs(np.real(np.vdot(psi, Hodd @ psi)) + 3 * len(ans.odd)) < 1e-10


def test_adjoint_gradient():
    for L, P in ((6, 3), (8, 3)):
        H = xxx_hamiltonian(L)
        ans = HVA(L, P, *xxx_bonds(L))
        p = np.random.default_rng(3).normal(0, 0.4, ans.n_params)
        e, _ = make_energy(H, ans)
        E_ad, g_ad = ans.energy_and_grad(p, H)
        assert abs(E_ad - e(p)) < 1e-12
        assert np.abs(central_grad(e)(p) - g_ad).max() < 1e-6


def test_hva_is_exact_for_L6():
    """Three layers reach the exact ground state at L=6 (report, Table 1)."""
    L, H = 6, xxx_hamiltonian(6)
    E0 = ground_state(H, k=1, sector=sz_sector_indices(L, L // 2))[0][0]
    import scipy.optimize as opt
    ans = HVA(L, 3, *xxx_bonds(L))
    rng = np.random.default_rng(0)
    best = min(
        opt.minimize(lambda q: ans.energy_and_grad(q, H), rng.uniform(-1, 1, 6),
                     jac=True, method="L-BFGS-B",
                     options={"ftol": 1e-15, "gtol": 1e-12}).fun
        for _ in range(10)
    )
    assert abs(best - E0) < 1e-8


# --------------------------------------------------------------- model 2
def test_hs_sin_identity():
    for L in (4, 6, 8, 12):
        for j in range(L):
            for k in range(j + 1, L):
                zj, zk = np.exp(2j * np.pi * j / L), np.exp(2j * np.pi * k / L)
                lhs = abs(zj - zk) ** 2
                rhs = 4 * np.sin(np.pi * (j - k) / L) ** 2
                assert abs(lhs - rhs) < 1e-12


def test_hs_operator_forms_agree():
    for L in (4, 6, 8):
        A = hs_hamiltonian(L, use_identity=True)
        B = hs_hamiltonian(L, use_identity=False)
        assert abs(A - B).max() < 1e-12


def test_hs_closed_form_energy():
    """E0 = -(pi^2/24)(L + 5/L), verified rather than assumed."""
    for L in (4, 6, 8, 10, 12):
        sec = sz_sector_indices(L, L // 2)
        E0 = ground_state(hs_hamiltonian(L), k=1, sector=sec)[0][0]
        assert abs(E0 - hs_exact_energy(L)) < 1e-9


def test_jastrow_is_the_ground_state():
    from run_model2 import jastrow_ground_state
    for L in (4, 6, 8, 10):
        sec = sz_sector_indices(L, L // 2)
        Hs = hs_hamiltonian(L)[sec][:, sec]
        psi, _ = jastrow_ground_state(L, extra_phase=True)
        Hpsi = Hs @ psi
        e = complex(np.vdot(psi, Hpsi)).real
        assert np.linalg.norm(Hpsi - e * psi) < 1e-9      # it is an eigenvector
        assert abs(e - hs_exact_energy(L)) < 1e-9         # and the lowest one


# --------------------------------------------------------------- model 3
def test_majorana_algebra():
    for N in (4, 6, 8, 10):
        assert check_clifford_algebra(N)


def test_majorana_squares_to_one():
    for N in (4, 8):
        for t in majorana_terms(N):
            assert set(t.s) <= set("IXYZ")
            M = pauli_string(t.s) * t.c
            assert abs(M @ M - sp.identity(M.shape[0], dtype=complex)).max() < 1e-12


def test_pauli_triplets_match_kron():
    for s in ("XYZI", "ZZXY", "IIIX", "YYYY", "XZYIZ"):
        r, c, v = pauli_string_triplets(s, 1.0)
        A = sp.coo_matrix((v, (r, c)), shape=(2 ** len(s),) * 2).toarray()
        assert np.abs(A - pauli_string(s).toarray()).max() < 1e-13


def test_syk_hermitian_and_traceless():
    for N in (8, 12):
        H, _, _ = syk_hamiltonian(N, seed=5)
        assert abs(H - H.getH()).max() < 1e-12
        assert abs(H.diagonal().sum()) < 1e-10


def test_commutation_predicates():
    assert commute("XX", "YY")          # differ on 2 sites -> commute
    assert not commute("XI", "YI")      # differ on 1 site  -> anticommute
    assert qubit_wise_commute("XI", "XZ")
    assert not qubit_wise_commute("XI", "YZ")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in fns:
        f()
        print(f"  ok  {f.__name__}")
    print(f"\n{len(fns)} tests passed")
