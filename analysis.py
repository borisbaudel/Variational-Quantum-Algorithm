"""
analysis.py -- three additions that turn statements in the report into
measurements.

1. `shift_rule_*` : the generalised parameter-shift rule for the exchange gate.

   For U(theta) = exp(-i theta G), the energy E(theta) is a trigonometric
   polynomial whose frequencies are the distinct *gaps* of the spectrum of G.
   For a single bond, G = sigma.sigma has eigenvalues +1 (triplet, threefold)
   and -3 (singlet), so the only non-zero gap is 4 and

       E(theta) = a + b cos(4 theta) + c sin(4 theta),

   which a *rescaled two-term* rule differentiates exactly:

       dE/dtheta = 2 [ E(theta + pi/8) - E(theta - pi/8) ].

   For a whole HVA sub-layer, G = sum over R disjoint bonds, whose eigenvalues
   are sums of per-bond ones; the gaps are then 4, 8, ..., 4R and a 2R-term
   rule is required.  Both statements are verified numerically, including the
   *failure* of the two-term rule for R >= 2 -- which is the reason the code
   uses adjoint differentiation instead.

2. `sample_energy_qwc` : an honest shot-noise estimator.  Pauli strings are
   packed into qubit-wise-commuting groups; each group is measured by rotating
   every qubit into the required basis and drawing bitstrings from |psi|^2, so
   that all strings in a group are estimated from the *same* shots, with their
   true correlations.  This is what makes the measurement-grouping numbers in
   the report operational rather than combinatorial.

3. `q2_majorana_*` : the free-fermion (q = 2) Majorana model, whose ground
   state is Gaussian and exactly obtainable, used as a warm start for SYK.
"""

from __future__ import annotations

import itertools

import numpy as np
import scipy.sparse as sp

from .core import pauli_string
from .syk import majorana_terms, pauli_string_triplets

# ---------------------------------------------------------------------------
# 1. parameter-shift rules
# ---------------------------------------------------------------------------
def shift_rule_two_term(energy_fn, theta: float, omega: float = 4.0) -> float:
    """(Omega/2) [ E(theta + pi/(2 Omega)) - E(theta - pi/(2 Omega)) ].

    Exact when E has the single frequency Omega.  For sigma.sigma, Omega = 4
    and the shift is pi/8.
    """
    s = np.pi / (2 * omega)
    return 0.5 * omega * (energy_fn(theta + s) - energy_fn(theta - s))


def shift_rule_general(energy_fn, theta: float, freqs) -> float:
    """Exact derivative of a trigonometric polynomial with the given positive
    frequencies, from 2R equidistant evaluations (Wierichs et al., Quantum 6,
    677 (2022); the construction here is the plain Fourier one).

    E(theta) = a_0 + sum_r [ a_r cos(w_r theta) + b_r sin(w_r theta) ].
    Sampling E at 2R+1 equidistant points on a period determined by the
    frequencies and differentiating the interpolant is numerically equivalent
    and much easier to get right than the closed-form coefficients, so that is
    what is done: the interpolation is exact, hence so is the derivative.
    """
    freqs = np.asarray(sorted(set(float(f) for f in freqs if f > 0)))
    R = len(freqs)
    M = 2 * R + 1
    # a common period: all frequencies here are integer multiples of freqs[0]
    T = 2 * np.pi / np.gcd.reduce(np.rint(freqs).astype(int)) if R else 2 * np.pi
    pts = theta + T * np.arange(M) / M
    vals = np.array([energy_fn(p) for p in pts])
    # least-squares fit of the exact model, then differentiate analytically
    cols = [np.ones(M)]
    for w in freqs:
        cols += [np.cos(w * pts), np.sin(w * pts)]
    A = np.vstack(cols).T
    coef, *_ = np.linalg.lstsq(A, vals, rcond=None)
    d = 0.0
    for r, w in enumerate(freqs):
        a, b = coef[1 + 2 * r], coef[2 + 2 * r]
        d += -a * w * np.sin(w * theta) + b * w * np.cos(w * theta)
    return float(d)


def sublayer_frequencies(n_bonds: int):
    """Gaps of sum_{r=1}^{R} (sigma.sigma)_r : the per-bond spectrum is
    {+1, -3}, so eigenvalues of the sum are R - 4k for k = 0..R and the
    distinct positive gaps are 4, 8, ..., 4R."""
    return [4.0 * k for k in range(1, n_bonds + 1)]


# ---------------------------------------------------------------------------
# 2. shot-noise estimation with qubit-wise-commuting groups
# ---------------------------------------------------------------------------
def qwc_groups(strings):
    """Greedy packing of Pauli strings into qubit-wise-commuting groups.

    Returns a list of (group_basis, members) where `group_basis` is a string in
    {I,X,Y,Z}^n giving, for each qubit, the basis every member requires (I if
    no member acts there).
    """
    groups = []
    for s in strings:
        placed = False
        for g in groups:
            basis = g["basis"]
            ok = all(b == "I" or c == "I" or b == c for b, c in zip(basis, s))
            if ok:
                g["basis"] = "".join(c if b == "I" else b for b, c in zip(basis, s))
                g["members"].append(s)
                placed = True
                break
        if not placed:
            groups.append({"basis": s, "members": [s]})
    return groups


_H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
_SDGH = _H @ np.array([[1, 0], [0, -1j]], dtype=complex)  # H S^dagger : Y -> Z


def _rotate_to_z(psi: np.ndarray, basis: str) -> np.ndarray:
    """Apply the single-qubit Cliffords that map each requested basis to Z."""
    from .ansatz import apply_1q

    n = len(basis)
    out = psi
    for q, b in enumerate(basis):
        if b == "X":
            out = apply_1q(out, _H, q, n)
        elif b == "Y":
            out = apply_1q(out, _SDGH, q, n)
    return out


def sample_energy_qwc(psi: np.ndarray, terms: dict, shots_per_group: int,
                      rng) -> float:
    """Shot-noise estimate of <psi|H|psi> for H = sum_s c_s P_s.

    All members of a qubit-wise-commuting group are estimated from the same
    bitstrings, which is exactly the saving that grouping is supposed to buy.
    """
    strings = [s for s, c in terms.items() if abs(c) > 1e-15]
    n = len(strings[0])
    total = 0.0
    for g in qwc_groups(strings):
        rot = _rotate_to_z(psi, g["basis"])
        p = np.abs(rot) ** 2
        p = p / p.sum()
        draws = rng.choice(p.size, size=shots_per_group, p=p)
        bits = ((draws[:, None] >> np.arange(n - 1, -1, -1)[None, :]) & 1)
        for s in g["members"]:
            support = np.array([k for k, ch in enumerate(s) if ch != "I"])
            par = bits[:, support].sum(axis=1) & 1
            est = float(np.mean(np.where(par, -1.0, 1.0)))
            total += terms[s].real * est
    return total


def shots_for_precision(terms: dict, epsilon: float) -> float:
    """Worst-case shot budget  ( sum_s |c_s| )^2 / epsilon^2  (ungrouped)."""
    l1 = sum(abs(c) for c in terms.values())
    return float(l1**2 / epsilon**2)


# ---------------------------------------------------------------------------
# 3. the free-fermion (q = 2) Majorana model, for warm starts
# ---------------------------------------------------------------------------
def q2_hamiltonian(N: int, J: float = 1.0, seed: int = 0):
    """H_2 = (i/2) sum_{i<j} K_ij chi_i chi_j with K real antisymmetric.

    The factor of i makes each term Hermitian: (i chi_i chi_j)^dagger
    = -i chi_j chi_i = +i chi_i chi_j for i != j.  The couplings are scaled so
    that the spectral width is comparable to the SYK model's, which is what
    makes the interpolation H(s) = (1-s) H_2 + s H_SYK well conditioned.
    """
    rng = np.random.default_rng(seed)
    n = N // 2
    chi = majorana_terms(N)
    sigma = np.sqrt(2.0 * J**2 / N)
    R, C, V = [], [], []
    for i in range(N):
        for j in range(i + 1, N):
            K = rng.normal(0.0, sigma)
            t = chi[i] * chi[j]
            r, c, v = pauli_string_triplets(t.s, 0.5j * K * t.c)
            R.append(r); C.append(c); V.append(v)
    H = sp.coo_matrix((np.concatenate(V), (np.concatenate(R), np.concatenate(C))),
                      shape=(2**n, 2**n), dtype=complex).tocsr()
    H.sum_duplicates()
    return H


def q2_ground_state(N: int, J: float = 1.0, seed: int = 0):
    """Exact ground state of the q = 2 model (a Gaussian state).

    At the sizes used here direct diagonalisation is cheaper than building the
    Bogoliubov transformation, and it is exact either way; the point of the
    construction is that on hardware this state is preparable in O(n^2) gates
    by a Givens-rotation network, not that it is hard to obtain classically.
    """
    H2 = q2_hamiltonian(N, J, seed)
    w, v = np.linalg.eigh(H2.toarray())
    return H2, float(w[0]), v[:, 0]
