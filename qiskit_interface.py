"""
qiskit_interface.py -- the same three Hamiltonians and the HVA circuit expressed
in Qiskit, together with a cross-validation against the in-house state-vector
engine.

Why both?  The in-house engine in `ansatz.py` exploits the block structure of
exp(-i theta sigma.sigma) and reaches L = 16 in seconds, which Qiskit's generic
simulator does not.  Qiskit is used here to prove that the fast engine
implements the circuit that it claims to implement, and to provide the
transpiled circuits (depth, two-qubit gate count) that one would actually run
on hardware.

Endianness.  core.py puts site 0 in the most significant tensor factor; Qiskit
puts qubit 0 in the least significant one.  All Pauli labels handed to Qiskit
are therefore reversed.  Because both the Hamiltonian and the ansatz are built
with the same reversal, expectation values agree exactly -- which is what the
cross-check verifies.

The exchange gate is exact in terms of standard rotations:
    exp(-i theta (XX + YY + ZZ)) = RXX(2 theta) RYY(2 theta) RZZ(2 theta),
since the three two-qubit Pauli products commute pairwise.
"""

from __future__ import annotations

import numpy as np

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp, Statevector

from .core import hs_coupling, xxx_bonds


# ---------------------------------------------------------------------------
# Hamiltonians as SparsePauliOp
# ---------------------------------------------------------------------------
def _pair_label(L: int, i: int, j: int, p: str) -> str:
    lab = ["I"] * L
    lab[i] = lab[j] = p
    return "".join(reversed(lab))  # reversed -> Qiskit little-endian


def xxx_sparse_pauli(L: int, J: float = 1.0) -> SparsePauliOp:
    terms = []
    for i in range(L):
        j = (i + 1) % L
        for p in "XYZ":
            terms.append((_pair_label(L, i, j, p), J))
    return SparsePauliOp.from_list(terms).simplify()


def hs_sparse_pauli(L: int, J: float = 1.0) -> SparsePauliOp:
    terms = []
    for i in range(L):
        for j in range(i + 1, L):
            c = hs_coupling(L, i, j, J)
            for p in "XYZ":
                terms.append((_pair_label(L, i, j, p), c))
    return SparsePauliOp.from_list(terms).simplify()


def syk_sparse_pauli(N: int, agg: dict) -> SparsePauliOp:
    """`agg` is the {pauli_string: coefficient} dict returned by syk_hamiltonian."""
    return SparsePauliOp.from_list(
        [("".join(reversed(s)), c) for s, c in agg.items() if abs(c) > 1e-15]
    ).simplify()


# ---------------------------------------------------------------------------
# the HVA as a Qiskit circuit
# ---------------------------------------------------------------------------
def singlet_prep(qc: QuantumCircuit, a: int, b: int) -> None:
    """|01> - |10> over sqrt(2) on qubits (a, b), from |00>."""
    qc.x(a)
    qc.h(a)
    qc.cx(a, b)
    qc.x(b)
    # |00> -> X_a -> |10>_a=1 -> H_a -> (|0>-|1>)/r2 (x) |0> -> CX -> (|00>-|11>)/r2
    # -> X_b -> (|01>-|10>)/r2      (in the qiskit ordering q_a q_b)


def exchange_block(qc: QuantumCircuit, theta: float, a: int, b: int) -> None:
    qc.rxx(2 * theta, a, b)
    qc.ryy(2 * theta, a, b)
    qc.rzz(2 * theta, a, b)


def hva_circuit(L: int, layers: int, params) -> QuantumCircuit:
    odd, even = xxx_bonds(L)
    qc = QuantumCircuit(L)
    for (a, b) in odd:
        singlet_prep(qc, L - 1 - a, L - 1 - b)
    for p in range(layers):
        for (a, b) in odd:
            exchange_block(qc, params[2 * p], L - 1 - a, L - 1 - b)
        for (a, b) in even:
            exchange_block(qc, params[2 * p + 1], L - 1 - a, L - 1 - b)
    return qc


# ---------------------------------------------------------------------------
# cross-validation
# ---------------------------------------------------------------------------
def crosscheck(L: int, layers: int, seed: int = 0):
    """Compare Qiskit and the in-house engine at random parameters.

    Returns (E_inhouse, E_qiskit, state_overlap, circuit_metrics).
    """
    from .ansatz import HVA
    from .core import xxx_hamiltonian

    rng = np.random.default_rng(seed)
    params = rng.normal(0.0, 0.6, 2 * layers)

    # in-house
    H = xxx_hamiltonian(L)
    ans = HVA(L, layers, *xxx_bonds(L))
    psi = ans.state(params)
    E_mine = float(np.real(np.vdot(psi, H @ psi)))

    # qiskit
    qc = hva_circuit(L, layers, params)
    sv = Statevector.from_instruction(qc)
    Hq = xxx_sparse_pauli(L)
    E_qk = float(np.real(sv.expectation_value(Hq)))

    # the two state vectors live in reversed orderings; reverse one of them
    v_mine = psi.reshape((2,) * L).transpose(*range(L - 1, -1, -1)).reshape(-1)
    overlap = abs(np.vdot(v_mine, np.asarray(sv.data)))

    tq = transpile(qc, basis_gates=["cx", "rz", "sx", "x"], optimization_level=2)
    metrics = {
        "depth_logical": qc.depth(),
        "depth_transpiled_cx_basis": tq.depth(),
        "n_cx": tq.count_ops().get("cx", 0),
        "n_params": 2 * layers,
    }
    return E_mine, E_qk, float(overlap), metrics
