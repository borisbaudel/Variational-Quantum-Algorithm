"""
vqe.py -- the variational loop.

Energies are evaluated exactly as <psi|H|psi> on the full state vector (no
shot noise).  This is deliberate: the questions ask for the *variational*
error with respect to exact diagonalisation, and mixing in sampling noise
would confuse the two sources of error.  Section 5 of the report discusses
what changes with finite sampling.

Gradients use central finite differences.  The parameter count is small
(2P for the HVA, n(P+1) for the brickwork) so this is cheaper than it looks
and avoids the subtlety that exp(-i theta sigma.sigma) has three distinct
eigenvalue gaps, for which the two-term parameter-shift rule does not hold.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.optimize as opt


@dataclass
class VQEResult:
    energy: float
    params: np.ndarray
    n_evals: int
    history: list = field(default_factory=list)
    success: bool = True


def make_energy(H, ansatz):
    counter = {"n": 0}

    def energy(params):
        counter["n"] += 1
        psi = ansatz.state(np.asarray(params, dtype=float))
        return float(np.real(np.vdot(psi, H @ psi)))

    return energy, counter


def central_grad(f, eps=1e-6):
    def grad(params):
        p = np.asarray(params, dtype=float)
        g = np.empty_like(p)
        for i in range(p.size):
            q = p.copy()
            q[i] = p[i] + eps
            fp = f(q)
            q[i] = p[i] - eps
            fm = f(q)
            g[i] = (fp - fm) / (2 * eps)
        return g

    return grad


def run_vqe(H, ansatz, n_restarts=8, seed=0, maxiter=2000, verbose=False) -> VQEResult:
    """Multi-start L-BFGS-B.  Random restarts are the cheapest defence against
    the local minima that the HVA landscape still contains at large L."""
    rng = np.random.default_rng(seed)
    best = None
    energy, counter = make_energy(H, ansatz)
    grad = central_grad(energy)

    for r in range(n_restarts):
        x0 = ansatz.init_params(rng)
        hist = []

        def cb(xk):
            hist.append(energy(xk))

        res = opt.minimize(
            energy, x0, jac=grad, method="L-BFGS-B",
            options={"maxiter": maxiter, "ftol": 1e-14, "gtol": 1e-10},
            callback=cb,
        )
        if best is None or res.fun < best.energy:
            best = VQEResult(float(res.fun), res.x.copy(), counter["n"], hist, bool(res.success))
        if verbose:
            print(f"    restart {r}: E = {res.fun:+.10f}")

    best.n_evals = counter["n"]
    return best


def gradient_variance(H, ansatz_factory, n_samples=200, seed=0):
    """Variance of dE/dtheta_0 over random parameter initialisations.

    An exponential decay of this variance with system size is the standard
    diagnostic of a barren plateau (McClean et al., Nat. Commun. 9, 4812).
    """
    rng = np.random.default_rng(seed)
    ansatz = ansatz_factory()
    energy, _ = make_energy(H, ansatz)
    eps = 1e-5
    vals = []
    for _ in range(n_samples):
        p = ansatz.init_params(rng)
        q = p.copy()
        q[0] = p[0] + eps
        fp = energy(q)
        q[0] = p[0] - eps
        fm = energy(q)
        vals.append((fp - fm) / (2 * eps))
    vals = np.asarray(vals)
    return float(vals.var()), float(np.abs(vals).mean())
