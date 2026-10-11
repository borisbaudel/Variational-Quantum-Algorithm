"""
syk.py -- the Sachdev-Ye-Kitaev model: Jordan-Wigner mapping, Hamiltonian
construction, Pauli bookkeeping and measurement grouping.

Mapping (N Majoranas -> n = N/2 qubits), with 0-based labels chi_0 ... chi_{N-1}:

    chi_{2k}   = ( Z_0 Z_1 ... Z_{k-1} )  X_k
    chi_{2k+1} = ( Z_0 Z_1 ... Z_{k-1} )  Y_k        k = 0 ... n-1

One checks directly that chi_a^2 = 1 and {chi_a, chi_b} = 2 delta_ab:
the Z-string makes operators on different qubits anticommute, and X_k, Y_k
anticommute on the same qubit.  See the report for the explicit verification,
which is also performed numerically by `check_clifford_algebra`.

Each quartic term chi_i chi_j chi_k chi_l is therefore a *single* Pauli string
with a phase, so H is a sum of exactly C(N,4) Pauli strings on N/2 qubits.
That combinatorial growth is the crux of Model 3.
"""

from __future__ import annotations

import itertools

import numpy as np
import scipy.sparse as sp

from .core import pauli_string

# ---------------------------------------------------------------------------
# symbolic Pauli-string algebra (phase in {+-1, +-i})
# ---------------------------------------------------------------------------
_MUL = {
    ("I", "I"): ("I", 1), ("I", "X"): ("X", 1), ("I", "Y"): ("Y", 1), ("I", "Z"): ("Z", 1),
    ("X", "I"): ("X", 1), ("X", "X"): ("I", 1), ("X", "Y"): ("Z", 1j), ("X", "Z"): ("Y", -1j),
    ("Y", "I"): ("Y", 1), ("Y", "X"): ("Z", -1j), ("Y", "Y"): ("I", 1), ("Y", "Z"): ("X", 1j),
    ("Z", "I"): ("Z", 1), ("Z", "X"): ("Y", 1j), ("Z", "Y"): ("X", -1j), ("Z", "Z"): ("I", 1),
}


class PauliTerm:
    """A Pauli string with a complex coefficient."""

    __slots__ = ("s", "c")

    def __init__(self, s: str, c: complex = 1.0):
        self.s, self.c = s, complex(c)

    def __mul__(self, other: "PauliTerm") -> "PauliTerm":
        out, phase = [], self.c * other.c
        for a, b in zip(self.s, other.s):
            ch, ph = _MUL[(a, b)]
            out.append(ch)
            phase *= ph
        return PauliTerm("".join(out), phase)

    def weight(self) -> int:
        return sum(ch != "I" for ch in self.s)

    def __repr__(self):
        return f"({self.c:+.4g})·{self.s}"


def majorana_terms(N: int) -> list[PauliTerm]:
    """The N Majorana operators as Pauli strings on n = N/2 qubits."""
    assert N % 2 == 0, "N must be even"
    n = N // 2
    ops = []
    for k in range(n):
        head = "Z" * k
        tail = "I" * (n - k - 1)
        ops.append(PauliTerm(head + "X" + tail, 1.0))  # chi_{2k}
        ops.append(PauliTerm(head + "Y" + tail, 1.0))  # chi_{2k+1}
    return ops


def check_clifford_algebra(N: int, tol: float = 1e-12) -> bool:
    """Numerically verify {chi_a, chi_b} = 2 delta_ab for the mapped operators."""
    n = N // 2
    mats = [pauli_string(t.s) * t.c for t in majorana_terms(N)]
    eye = sp.identity(2**n, format="csr", dtype=complex)
    for a in range(N):
        for b in range(N):
            anti = mats[a] @ mats[b] + mats[b] @ mats[a]
            target = 2 * eye if a == b else None
            err = abs(anti - target).max() if a == b else abs(anti).max()
            if err > tol:
                return False
    return True


# ---------------------------------------------------------------------------
# the Hamiltonian
# ---------------------------------------------------------------------------
def syk_couplings(N: int, J: float = 1.0, seed: int = 0) -> dict:
    """Gaussian couplings with zero mean and variance <J^2> = 6 J^2 / N^3,
    drawn once for each ordered quadruple i<j<k<l (antisymmetry then fixes the
    rest, and the 1/4! in the assignment's definition exactly undoes the
    4! orderings)."""
    rng = np.random.default_rng(seed)
    sigma = np.sqrt(6.0 * J**2 / N**3)
    return {q: rng.normal(0.0, sigma) for q in itertools.combinations(range(N), 4)}


def syk_pauli_terms(N: int, couplings: dict) -> list[PauliTerm]:
    """H as a list of Pauli strings.  Exactly C(N,4) of them."""
    chi = majorana_terms(N)
    terms = []
    for (i, j, k, l), Jc in couplings.items():
        t = chi[i] * chi[j] * chi[k] * chi[l]
        terms.append(PauliTerm(t.s, Jc * t.c))
    return terms


def pauli_string_triplets(s: str, coeff: complex):
    """(rows, cols, vals) of a Pauli string, built directly in O(2^n).

    A Pauli string is a signed permutation matrix.  Acting on a basis state
    |b> with bit b_k at site k (site 0 = most significant, matching core.py):

        X_k : flips b_k, factor 1
        Y_k : flips b_k, factor  i (-1)^{b_k}        since Y|b> = i(-1)^b |1-b>
        Z_k : no flip,  factor    (-1)^{b_k}
        I_k : no flip,  factor 1

    so the column index i maps to row j = i XOR flipmask with amplitude
    i^{n_Y} (-1)^{popcount(i & (ymask|zmask))}.  Assembling all terms this way
    and calling coo_matrix once is orders of magnitude faster than repeatedly
    adding sparse matrices, which is what makes N = 20 (4845 terms) tractable.
    """
    n = len(s)
    flip = 0
    signmask = 0
    n_y = 0
    for k, ch in enumerate(s):
        bit = 1 << (n - 1 - k)
        if ch == "X":
            flip |= bit
        elif ch == "Y":
            flip |= bit
            signmask |= bit
            n_y += 1
        elif ch == "Z":
            signmask |= bit
    dim = 2**n
    cols = np.arange(dim, dtype=np.int64)
    rows = cols ^ flip
    par = np.zeros(dim, dtype=np.int64)
    m = signmask
    while m:
        b = m & (-m)
        par += (cols & b) != 0
        m ^= b
    vals = coeff * (1j**n_y) * np.where(par & 1, -1.0, 1.0)
    return rows, cols, vals


def syk_hamiltonian(N: int, J: float = 1.0, seed: int = 0):
    """Sparse 2^{N/2} x 2^{N/2} Hamiltonian, plus its Pauli decomposition."""
    couplings = syk_couplings(N, J, seed)
    terms = syk_pauli_terms(N, couplings)
    n = N // 2
    # group identical strings first -- cheaper and exposes cancellations
    agg: dict[str, complex] = {}
    for t in terms:
        agg[t.s] = agg.get(t.s, 0.0) + t.c
    R, C, V = [], [], []
    for s, c in agg.items():
        if abs(c) > 1e-15:
            r, cc, v = pauli_string_triplets(s, c)
            R.append(r); C.append(cc); V.append(v)
    H = sp.coo_matrix(
        (np.concatenate(V), (np.concatenate(R), np.concatenate(C))),
        shape=(2**n, 2**n), dtype=complex,
    ).tocsr()
    H.sum_duplicates()
    return H, terms, agg


# ---------------------------------------------------------------------------
# measurement grouping
# ---------------------------------------------------------------------------
def commute(a: str, b: str) -> bool:
    """Two Pauli strings commute iff they differ on an even number of qubits
    where both act non-trivially."""
    d = sum(1 for x, y in zip(a, b) if x != "I" and y != "I" and x != y)
    return d % 2 == 0


def qubit_wise_commute(a: str, b: str) -> bool:
    return all(x == "I" or y == "I" or x == y for x, y in zip(a, b))


def group_strings(strings: list[str], mode: str = "commute") -> int:
    """Number of simultaneously measurable groups from a greedy colouring of
    the non-commutation graph (largest-first / Welsh-Powell)."""
    import networkx as nx

    pred = commute if mode == "commute" else qubit_wise_commute
    G = nx.Graph()
    G.add_nodes_from(range(len(strings)))
    for i in range(len(strings)):
        for j in range(i + 1, len(strings)):
            if not pred(strings[i], strings[j]):
                G.add_edge(i, j)
    colouring = nx.greedy_color(G, strategy="largest_first")
    return (max(colouring.values()) + 1) if colouring else 0
