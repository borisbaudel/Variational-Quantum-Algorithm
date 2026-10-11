"""
ansatz.py -- state-vector simulation of the variational circuits.

Two ansaetze are provided:

1. `HVA` -- the Hamiltonian Variational Ansatz for the XXX chain.  The PBC chain
   is split into two layers of *disjoint* bonds, so each layer exponential
   exp(-i theta H_layer) factorises exactly into commuting two-site gates.
   There is therefore **no Trotter error**: the circuit is implemented exactly.
   The reference state is the product of nearest-neighbour singlets, which is
   the exact ground state of the odd layer and already an SU(2) singlet with
   S^z_tot = 0.  Because every gate exp(-i theta sigma.sigma) is an SU(2)
   scalar, the whole ansatz stays inside the total-singlet, S^z = 0 sector --
   exactly where the antiferromagnetic ground state lives.  This symmetry
   protection is what makes the optimisation tractable at L = 16.

2. `HardwareEfficient` -- the usual RY + CZ brickwork.  It carries no symmetry
   and is used deliberately as the *negative control* for the SYK model.
"""

from __future__ import annotations

import numpy as np
import scipy.linalg as la

from .core import exchange_op

# ---------------------------------------------------------------------------
# exact 4x4 gate for exp(-i theta sigma.sigma)
# ---------------------------------------------------------------------------
_SS2 = np.array(
    [  # X@X + Y@Y + Z@Z on two qubits, in the basis |00>,|01>,|10>,|11>
        [1, 0, 0, 0],
        [0, -1, 2, 0],
        [0, 2, -1, 0],
        [0, 0, 0, 1],
    ],
    dtype=complex,
)


def exchange_gate(theta: float) -> np.ndarray:
    """exp(-i theta (XX+YY+ZZ)) as a 4x4 unitary.

    sigma.sigma has eigenvalue +1 on the triplet and -3 on the singlet, so this
    is exp(-i theta) on the triplet and exp(+3 i theta) on the singlet.
    """
    return la.expm(-1j * theta * _SS2)


SINGLET = np.array([0.0, 1.0, -1.0, 0.0], dtype=complex) / np.sqrt(2.0)


def apply_exchange(psi: np.ndarray, theta: float, a: int, b: int, L: int) -> np.ndarray:
    """Fast specialised version of exp(-i theta sigma_a.sigma_b).

    sigma.sigma is block diagonal: it is the identity on {|00>, |11>} and
    -I + 2X on {|01>, |10>}.  Hence

        exp(-i theta sigma.sigma) = e^{-i theta}            on |00>, |11>
                                  = e^{+i theta}( cos 2theta I - i sin 2theta X )
                                                            on |01>, |10>.

    Exploiting this avoids a general 4x4 contraction and is what brings the
    L = 16 optimisation down to a few minutes.  Verified against `apply_2q`
    in tests/test_core.py.
    """
    if a > b:
        a, b = b, a
    d0, d1, d2 = 2**a, 2 ** (b - a - 1), 2 ** (L - b - 1)
    t = psi.reshape(d0, 2, d1, 2, d2).copy()
    ph = np.exp(-1j * theta)
    al = np.exp(1j * theta) * np.cos(2 * theta)
    be = -1j * np.exp(1j * theta) * np.sin(2 * theta)
    u = t[:, 0, :, 1, :].copy()
    v = t[:, 1, :, 0, :].copy()
    t[:, 0, :, 0, :] *= ph
    t[:, 1, :, 1, :] *= ph
    t[:, 0, :, 1, :] = al * u + be * v
    t[:, 1, :, 0, :] = be * u + al * v
    return t.reshape(-1)


# ---------------------------------------------------------------------------
# state-vector primitives
# ---------------------------------------------------------------------------
def apply_2q(psi: np.ndarray, G: np.ndarray, a: int, b: int, L: int) -> np.ndarray:
    """Apply the 4x4 gate G to sites (a, b) of a state vector of L qubits.

    Implemented as a single contraction over the five-way reshape
    (left, site_a, middle, site_b, right), which keeps the arrays contiguous and
    is what makes L = 16 affordable.  The (a > b) case -- needed for the
    periodic bond (L-1, 0) -- is handled by transposing the gate.
    """
    G4 = G.reshape(2, 2, 2, 2)  # (out_a, out_b, in_a, in_b)
    if a > b:
        a, b = b, a
        G4 = G4.transpose(1, 0, 3, 2)
    d0, d1, d2 = 2**a, 2 ** (b - a - 1), 2 ** (L - b - 1)
    pr = psi.reshape(d0, 2, d1, 2, d2)
    out = np.einsum("stab,iajbk->isjtk", G4, pr, optimize=True)
    return out.reshape(-1)


def apply_1q(psi: np.ndarray, G: np.ndarray, a: int, L: int) -> np.ndarray:
    t = psi.reshape((2,) * L)
    t = np.tensordot(G, t, axes=([1], [a]))
    t = np.moveaxis(t, 0, a)
    return t.reshape(-1)


def ry(theta: float) -> np.ndarray:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


CZ = np.diag([1.0, 1.0, 1.0, -1.0]).astype(complex)


# ---------------------------------------------------------------------------
# ansatz 1 : Hamiltonian Variational Ansatz
# ---------------------------------------------------------------------------
class HVA:
    def __init__(self, L: int, layers: int, odd_layer, even_layer):
        self.L, self.P = L, layers
        self.odd, self.even = odd_layer, even_layer
        self.n_params = 2 * layers
        self._sublayers = [odd_layer, even_layer] * layers  # bonds of sub-layer k
        self._Hlayer = None  # lazily built sparse generators, for the gradient

    # -- sparse generators H_odd, H_even (needed only by the analytic gradient)
    def layer_generators(self):
        if self._Hlayer is None:
            import scipy.sparse as sp

            gens = []
            for bonds in (self.odd, self.even):
                M = sp.csr_matrix((2**self.L, 2**self.L), dtype=complex)
                for (a, b) in bonds:
                    M = M + exchange_op(self.L, a, b)
                gens.append(M.tocsr())
            self._Hlayer = gens
        return self._Hlayer

    def _apply_sublayer(self, psi, theta, bonds, dagger=False):
        th = -theta if dagger else theta
        for (a, b) in bonds:
            psi = apply_exchange(psi, th, a, b, self.L)
        return psi

    def energy_and_grad(self, params, H):
        """Exact energy and analytic gradient by adjoint (reverse-mode)
        differentiation.

        For a sub-layer U_k = exp(-i theta_k H_k) with H_k a sum of *disjoint*
        two-site terms, d/dtheta_k U_k = -i H_k U_k, hence

            dE/dtheta_k = 2 Re < lambda | (-i H_k) psi_{k+1} >,

        where psi_{k+1} is the state after sub-layer k and lambda is the
        back-propagated adjoint state.  The cost is about two forward passes
        instead of the 2 x n_params of a finite-difference gradient.
        """
        params = np.asarray(params, dtype=float)
        gens = self.layer_generators()
        # forward pass, storing the state after every sub-layer
        psi = self.reference()
        states = [psi]
        for k, bonds in enumerate(self._sublayers):
            psi = self._apply_sublayer(psi, params[k], bonds)
            states.append(psi)
        E = float(np.real(np.vdot(psi, H @ psi)))
        # backward pass
        lam = H @ psi
        grad = np.zeros_like(params)
        for k in range(len(self._sublayers) - 1, -1, -1):
            Hk = gens[k % 2]
            grad[k] = 2.0 * np.real(np.vdot(lam, -1j * (Hk @ states[k + 1])))
            lam = self._apply_sublayer(lam, params[k], self._sublayers[k], dagger=True)
        return E, grad

    def reference(self) -> np.ndarray:
        """Product of singlets on the odd-layer bonds."""
        psi = np.array([1.0 + 0j])
        for _ in self.odd:
            psi = np.kron(psi, SINGLET)
        # the kron above assumes the odd bonds are (0,1),(2,3),... which they are
        return psi

    def state(self, params: np.ndarray) -> np.ndarray:
        psi = self.reference()
        for k, bonds in enumerate(self._sublayers):
            psi = self._apply_sublayer(psi, params[k], bonds)
        return psi

    def init_params(self, rng, scale: float = 0.1) -> np.ndarray:
        return rng.normal(0.0, scale, self.n_params)


# ---------------------------------------------------------------------------
# ansatz 2 : hardware-efficient brickwork (negative control)
# ---------------------------------------------------------------------------
class HardwareEfficient:
    def __init__(self, n_qubits: int, layers: int):
        self.n, self.P = n_qubits, layers
        self.n_params = n_qubits * (layers + 1)

    def state(self, params: np.ndarray) -> np.ndarray:
        n = self.n
        psi = np.zeros(2**n, dtype=complex)
        psi[0] = 1.0
        k = 0
        for q in range(n):
            psi = apply_1q(psi, ry(params[k]), q, n)
            k += 1
        for _ in range(self.P):
            for q in range(0, n - 1, 2):
                psi = apply_2q(psi, CZ, q, q + 1, n)
            for q in range(1, n - 1, 2):
                psi = apply_2q(psi, CZ, q, q + 1, n)
            for q in range(n):
                psi = apply_1q(psi, ry(params[k]), q, n)
                k += 1
        return psi

    def init_params(self, rng, scale: float = np.pi) -> np.ndarray:
        return rng.uniform(-scale, scale, self.n_params)
