"""
core.py -- Pauli algebra, sparse Hamiltonian construction and exact diagonalisation
for the three spin-chain models of the assignment.

Conventions
-----------
* Sites are indexed 0..L-1 (the assignment uses 1..L; the shift is immaterial).
* Operators are built from *Pauli* matrices, NOT spin-1/2 operators.
  Since sigma = 2 S we have  sigma_i . sigma_j = 4 S_i . S_j.  This factor of 4
  matters when comparing against textbook Heisenberg / Haldane-Shastry results,
  and is handled explicitly in the report.
* Qubit ordering: the computational basis index b encodes site 0 as the MOST
  significant bit, i.e. |s_0 s_1 ... s_{L-1}>.  This is the natural "left to
  right" convention; note it is the reverse of Qiskit's little-endian order,
  and the conversion is done explicitly in qiskit_interface.py.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# ----------------------------------------------------------------------------
# single-site Pauli matrices
# ----------------------------------------------------------------------------
I2 = sp.identity(2, format="csr", dtype=complex)
SX = sp.csr_matrix(np.array([[0, 1], [1, 0]], dtype=complex))
SY = sp.csr_matrix(np.array([[0, -1j], [1j, 0]], dtype=complex))
SZ = sp.csr_matrix(np.array([[1, 0], [0, -1]], dtype=complex))

PAULI = {"I": I2, "X": SX, "Y": SY, "Z": SZ}


def pauli_string(label: str) -> sp.csr_matrix:
    """Sparse operator for a full Pauli string, e.g. 'XIZY' on 4 sites.

    Site 0 is the leftmost character and the most significant tensor factor.
    """
    out = PAULI[label[0]]
    for ch in label[1:]:
        out = sp.kron(out, PAULI[ch], format="csr")
    return out


def two_site_op(L: int, i: int, j: int, op_i: str, op_j: str) -> sp.csr_matrix:
    """Sparse operator with `op_i` on site i, `op_j` on site j, identity elsewhere."""
    label = ["I"] * L
    label[i] = op_i
    label[j] = op_j
    return pauli_string("".join(label))


def exchange_op(L: int, i: int, j: int) -> sp.csr_matrix:
    """sigma_i . sigma_j = X_i X_j + Y_i Y_j + Z_i Z_j  (sparse, 2^L x 2^L)."""
    return (
        two_site_op(L, i, j, "X", "X")
        + two_site_op(L, i, j, "Y", "Y")
        + two_site_op(L, i, j, "Z", "Z")
    )


# ----------------------------------------------------------------------------
# Model 1 : XXX Heisenberg chain, periodic boundary conditions
#           H = J sum_i ( X_i X_{i+1} + Y_i Y_{i+1} + Z_i Z_{i+1} )
# ----------------------------------------------------------------------------
def xxx_hamiltonian(L: int, J: float = 1.0) -> sp.csr_matrix:
    H = sp.csr_matrix((2**L, 2**L), dtype=complex)
    for i in range(L):
        H = H + J * exchange_op(L, i, (i + 1) % L)
    return H.tocsr()


def xxx_bonds(L: int):
    """Bond list for the PBC chain, split into the two commuting sub-layers
    used by the Hamiltonian Variational Ansatz.  Requires L even."""
    assert L % 2 == 0, "even L only"
    odd_layer = [(i, i + 1) for i in range(0, L, 2)]          # (0,1) (2,3) ...
    even_layer = [(i, (i + 1) % L) for i in range(1, L, 2)]   # (1,2) (3,4) ... (L-1,0)
    return odd_layer, even_layer


# ----------------------------------------------------------------------------
# Model 2 : Haldane-Shastry chain on the unit circle
#           H = (pi^2 J / L^2) sum_{j<k} sigma_j . sigma_k / |z_j - z_k|^2
#           with z_j = exp(2 pi i j / L)
#
# Using the identity |z_j - z_k|^2 = 4 sin^2( pi (j-k) / L )   (see report)
# this is the textbook Haldane-Shastry Hamiltonian.
# ----------------------------------------------------------------------------
def hs_coupling(L: int, j: int, k: int, J: float = 1.0, use_identity: bool = True) -> float:
    """Coefficient of sigma_j.sigma_k in H_HS."""
    if use_identity:
        denom = 4.0 * np.sin(np.pi * (j - k) / L) ** 2
    else:  # literal evaluation from the complex coordinates, used as a cross-check
        zj = np.exp(2j * np.pi * j / L)
        zk = np.exp(2j * np.pi * k / L)
        denom = abs(zj - zk) ** 2
    return (np.pi**2 * J / L**2) / denom


def hs_hamiltonian(L: int, J: float = 1.0, use_identity: bool = True) -> sp.csr_matrix:
    H = sp.csr_matrix((2**L, 2**L), dtype=complex)
    for j in range(L):
        for k in range(j + 1, L):
            H = H + hs_coupling(L, j, k, J, use_identity) * exchange_op(L, j, k)
    return H.tocsr()


def hs_exact_energy(L: int, J: float = 1.0) -> float:
    """Closed-form Haldane-Shastry ground-state energy, in the *Pauli*
    normalisation of this assignment.

    The textbook result for  H_S = J (pi/L)^2 sum_{j<k} S_j.S_k / sin^2(pi(j-k)/L)
    is   E_0 = -(pi^2 J / 24) (L + 5/L).
    The derivation in the report shows that the Hamiltonian of the assignment is
    exactly equal to H_S, so the same formula applies verbatim.

    This function is *verified numerically against exact diagonalisation* in
    run_model2.py -- it is not taken on trust.
    """
    return -(np.pi**2 * J / 24.0) * (L + 5.0 / L)


# ----------------------------------------------------------------------------
# total S^z sector projection (both models conserve S^z_total and total spin)
# ----------------------------------------------------------------------------
def sz_sector_indices(L: int, n_up: int) -> np.ndarray:
    """Basis indices with exactly `n_up` spins up (bit value 0 == up)."""
    idx = np.arange(2**L, dtype=np.int64)
    popcount = np.zeros_like(idx)
    for b in range(L):
        popcount += (idx >> b) & 1
    return np.nonzero(popcount == (L - n_up))[0]


def ground_state(H: sp.csr_matrix, k: int = 1, sector: np.ndarray | None = None):
    """Lowest eigenvalue(s) and eigenvector(s) of a sparse Hermitian H.

    If `sector` is given, H is first restricted to that subspace (a large
    speed-up: the S^z = 0 sector of L=16 has 12870 states instead of 65536).
    Dense LAPACK is used for small dimensions where Lanczos is unreliable.
    """
    M = H if sector is None else H[sector][:, sector]
    dim = M.shape[0]
    if dim <= 512:
        w, v = np.linalg.eigh(M.toarray())
        return w[:k], v[:, :k]
    w, v = spla.eigsh(M, k=k, which="SA", maxiter=20000, tol=0)
    order = np.argsort(w)
    return w[order], v[:, order]


def full_spectrum(H: sp.csr_matrix, sector: np.ndarray | None = None) -> np.ndarray:
    M = H if sector is None else H[sector][:, sector]
    return np.linalg.eigvalsh(M.toarray())
