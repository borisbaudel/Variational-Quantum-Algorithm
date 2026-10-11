"""
run_model3b.py -- Model 3, second pass: a fair negative control, and ADAPT-VQE.

Three things are fixed or added relative to run_model3.py.

1. The standard RY + CZ "hardware-efficient" ansatz produces only *real*
   amplitudes, while the Jordan-Wigner image of H_SYK has complex matrix
   elements.  Before blaming barren plateaus for its failure we must check
   whether the ground state is intrinsically complex -- otherwise part of the
   error is a trivial representability failure, not an optimisation one.
   `complexity_of_ground_state` measures exactly that.

2. A complex-capable brickwork (RY + RZ + CZ) is therefore added as the honest
   control.

3. ADAPT-VQE is implemented, with the operator pool taken from the Hamiltonian
   itself: the quartic Majorana terms chi_i chi_j chi_k chi_l, each of which is
   a single Pauli string P_m with P_m^2 = 1, so that

       exp(-i theta P_m) = cos(theta) 1 - i sin(theta) P_m

   is applied in two sparse matrix-vector products.  At theta = 0 the gradient
   of adding a new generator is

       dE/dtheta |_0 = < psi | i [P_m, H] | psi >,

   which is what selects the next operator.  Every new operator enters at
   theta = 0, so the construction is a warm start by design -- the same
   mechanism that gained an order of magnitude in Model 1.

Produces results/model3b.json
"""

import json
import time

import numpy as np
import scipy.optimize as opt
import scipy.sparse as sp

from spinvqe.ansatz import HardwareEfficient, apply_1q, apply_2q, ry, CZ
from spinvqe.core import pauli_string
from spinvqe.syk import syk_hamiltonian
from spinvqe.vqe import run_vqe


# --------------------------------------------------------------------------
def rz(theta):
    return np.diag([np.exp(-0.5j * theta), np.exp(0.5j * theta)]).astype(complex)


class HardwareEfficientComplex:
    """RY + RZ + CZ brickwork: same connectivity as `HardwareEfficient` but
    able to represent complex amplitudes."""

    def __init__(self, n_qubits, layers):
        self.n, self.P = n_qubits, layers
        self.n_params = 2 * n_qubits * (layers + 1)

    def state(self, params):
        n = self.n
        psi = np.zeros(2**n, dtype=complex)
        psi[0] = 1.0
        k = 0
        for _ in range(self.P + 1):
            for q in range(n):
                psi = apply_1q(psi, ry(params[k]), q, n); k += 1
                psi = apply_1q(psi, rz(params[k]), q, n); k += 1
            if k < self.n_params:
                for q in range(0, n - 1, 2):
                    psi = apply_2q(psi, CZ, q, q + 1, n)
                for q in range(1, n - 1, 2):
                    psi = apply_2q(psi, CZ, q, q + 1, n)
        return psi

    def init_params(self, rng, scale=np.pi):
        return rng.uniform(-scale, scale, self.n_params)


def complexity_of_ground_state(psi):
    """After removing the best global phase, how much imaginary part is left?

    Returns ||Im(e^{-i phi} psi)|| / ||psi|| minimised over phi.  Zero means the
    state is real up to a phase and a real-amplitude ansatz could in principle
    represent it.
    """
    # minimise over phi analytically: the optimal phi aligns the dominant
    # eigenvector of the 2x2 real form; a 1-D scan is quite sufficient here
    best = 1.0
    for phi in np.linspace(0, np.pi, 721):
        v = np.exp(-1j * phi) * psi
        best = min(best, float(np.linalg.norm(v.imag)))
    return best / float(np.linalg.norm(psi))


# --------------------------------------------------------------------------
class AdaptAnsatz:
    """psi = prod_k exp(-i theta_k P_k) |ref>, with P_k Hermitian Pauli strings."""

    def __init__(self, ops, ref):
        self.ops = list(ops)          # sparse Hermitian Pauli strings
        self.ref = ref
        self.n_params = len(self.ops)

    def _apply(self, psi, P, theta):
        return np.cos(theta) * psi - 1j * np.sin(theta) * (P @ psi)

    def state(self, params):
        psi = self.ref.copy()
        for P, th in zip(self.ops, params):
            psi = self._apply(psi, P, th)
        return psi

    def energy_and_grad(self, params, H):
        psi = self.ref.copy()
        states = [psi]
        for P, th in zip(self.ops, params):
            psi = self._apply(psi, P, th)
            states.append(psi)
        E = float(np.real(np.vdot(psi, H @ psi)))
        lam = H @ psi
        g = np.zeros(len(self.ops))
        for k in range(len(self.ops) - 1, -1, -1):
            P = self.ops[k]
            g[k] = 2.0 * np.real(np.vdot(lam, -1j * (P @ states[k + 1])))
            lam = self._apply(lam, P, -params[k])
        return E, g


def adapt_vqe(H, pool, ref, max_ops=40, grad_tol=1e-6, verbose=True):
    """Greedy ADAPT-VQE. `pool` is a list of sparse Hermitian Pauli strings."""
    chosen, params = [], np.zeros(0)
    psi = ref.copy()
    history = []
    for it in range(max_ops):
        # operator selection: dE/dtheta at theta=0 is <psi| i[P,H] |psi>
        Hpsi = H @ psi
        grads = []
        for P in pool:
            Ppsi = P @ psi
            # <psi| i(PH - HP) |psi> = 2 Im <P psi | H psi>
            grads.append(2.0 * np.imag(np.vdot(Ppsi, Hpsi)))
        grads = np.abs(np.asarray(grads))
        m = int(np.argmax(grads))
        if grads[m] < grad_tol:
            break
        chosen.append(pool[m])
        params = np.concatenate([params, [0.0]])
        ans = AdaptAnsatz(chosen, ref)
        res = opt.minimize(lambda p: ans.energy_and_grad(p, H), params, jac=True,
                           method="L-BFGS-B",
                           options={"maxiter": 800, "ftol": 1e-15, "gtol": 1e-12})
        params = res.x
        psi = ans.state(params)
        history.append({"n_ops": len(chosen), "E": float(res.fun),
                        "max_pool_grad": float(grads[m])})
        if verbose and (it < 5 or it % 5 == 4):
            print(f"        op {len(chosen):3d}: E = {res.fun:+.8f}  "
                  f"|grad|max = {grads[m]:.2e}", flush=True)
    return float(history[-1]["E"]) if history else float(np.real(np.vdot(ref, H @ ref))), params, history


# --------------------------------------------------------------------------
def main():
    out = {}
    rng_global = np.random.default_rng(0)

    for N in (8, 10, 12, 14):
        n = N // 2
        print(f"=== N = {N}  (n = {n} qubits)", flush=True)
        H, terms, agg = syk_hamiltonian(N, seed=100)
        Hd = H.toarray()
        w, v = np.linalg.eigh(Hd)
        E0 = float(w[0])
        gs = v[:, 0]
        cplx = complexity_of_ground_state(gs)
        deg = int(np.sum(w - w[0] < 1e-9))
        print(f"    E0 = {E0:+.8f}   degeneracy {deg}   "
              f"residual imaginary weight of the ground state: {cplx:.4f}", flush=True)

        rec = {"E0": E0, "degeneracy": deg, "gs_imaginary_weight": cplx,
               "n_terms": len(agg)}

        # ---- control 1: real-amplitude brickwork
        rec["real_brickwork"] = {}
        for P in (4, 12):
            ans = HardwareEfficient(n, P)
            res = run_vqe(H, ans, n_restarts=5, seed=3, maxiter=2500)
            rec["real_brickwork"][str(P)] = {
                "n_params": ans.n_params, "E": res.energy,
                "rel_err": abs((res.energy - E0) / E0)}
            print(f"    RY+CZ      P={P:2d} ({ans.n_params:3d} params): "
                  f"E = {res.energy:+.6f}   rel err = "
                  f"{100*abs((res.energy-E0)/E0):5.1f}%", flush=True)

        # ---- control 2: complex-capable brickwork
        rec["complex_brickwork"] = {}
        for P in (4, 12):
            ans = HardwareEfficientComplex(n, P)
            res = run_vqe(H, ans, n_restarts=5, seed=3, maxiter=2500)
            rec["complex_brickwork"][str(P)] = {
                "n_params": ans.n_params, "E": res.energy,
                "rel_err": abs((res.energy - E0) / E0)}
            print(f"    RY+RZ+CZ   P={P:2d} ({ans.n_params:3d} params): "
                  f"E = {res.energy:+.6f}   rel err = "
                  f"{100*abs((res.energy-E0)/E0):5.1f}%", flush=True)

        # ---- ADAPT-VQE with the Hamiltonian's own terms as the pool
        pool_strings = sorted(agg, key=lambda s: -abs(agg[s]))[:min(300, len(agg))]
        pool = [pauli_string(s) for s in pool_strings]

        def product_ref(complex_phases: bool):
            r = np.zeros(2**n, dtype=complex)
            r[0] = 1.0
            for q in range(n):
                r = apply_1q(r, ry(0.7 + 0.13 * q), q, n)
                if complex_phases:
                    r = apply_1q(r, rz(0.4 + 0.21 * q), q, n)
            return r

        print(f"    ADAPT-VQE, pool of {len(pool)} quartic terms", flush=True)
        rec["adapt"] = {}
        for tag, cplx_ref in (("real_reference", False), ("complex_reference", True)):
            t0 = time.time()
            ref = product_ref(cplx_ref)
            E_ad, p_ad, hist = adapt_vqe(H, pool, ref,
                                         max_ops=min(60, 6 * n), verbose=cplx_ref)
            rec["adapt"][tag] = {
                "pool_size": len(pool), "n_ops": int(len(p_ad)),
                "E": E_ad, "rel_err": abs((E_ad - E0) / E0),
                "history": hist, "t": time.time() - t0}
            print(f"    -> ADAPT ({tag}): E = {E_ad:+.8f} with {len(p_ad)} operators"
                  f"   rel err = {100*abs((E_ad-E0)/E0):6.2f}%   "
                  f"({time.time()-t0:.1f}s)", flush=True)
        print(flush=True)

        out[str(N)] = rec
        with open("results/model3b.json", "w") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
