"""
run_model1.py -- Model 1: XXX Heisenberg chain with periodic boundary conditions.

Produces:
  results/model1.json
  figures/model1_error_vs_layers.pdf
  figures/model1_finite_size.pdf
"""

import json
import time

import numpy as np

from spinvqe.ansatz import HVA
from spinvqe.core import ground_state, sz_sector_indices, xxx_bonds, xxx_hamiltonian
from spinvqe.vqe import VQEResult

import scipy.optimize as opt

# exact thermodynamic limit, Bethe ansatz, in the Pauli normalisation:
#   H = J sum sigma.sigma = 4 J sum S.S   and   E_S/L -> J(1/4 - ln 2)
#   => E/L -> 4 J (1/4 - ln 2) = J (1 - 4 ln 2)
E_INF = 1.0 - 4.0 * np.log(2.0)

LS = [4, 6, 8, 12, 16]
LAYER_PLAN = {4: [1, 2, 3], 6: [1, 2, 3, 4], 8: [1, 2, 3, 4, 5],
              12: [1, 2, 3, 4, 6], 16: [2, 4, 6, 8]}
RESTARTS = {4: 6, 6: 6, 8: 5, 12: 3, 16: 2}
MAXITER = {4: 2000, 6: 2000, 8: 1500, 12: 800, 16: 400}


def vqe_adjoint(H, ansatz, n_restarts, maxiter, seed=0, verbose=True):
    """L-BFGS-B driven by the analytic adjoint gradient."""
    rng = np.random.default_rng(seed)
    best = None
    n_calls = {"n": 0}

    def fg(p):
        n_calls["n"] += 1
        E, g = ansatz.energy_and_grad(p, H)
        return E, g

    for r in range(n_restarts):
        x0 = ansatz.init_params(rng, scale=0.15 + 0.25 * r)
        res = opt.minimize(fg, x0, jac=True, method="L-BFGS-B",
                           options={"maxiter": maxiter, "ftol": 1e-15, "gtol": 1e-11})
        if best is None or res.fun < best.energy:
            best = VQEResult(float(res.fun), res.x.copy(), n_calls["n"], [], bool(res.success))
        if verbose:
            print(f"      restart {r}: E = {res.fun:+.10f}  ({res.nit} iters)", flush=True)
    best.n_evals = n_calls["n"]
    return best


def main():
    out = {"E_inf_exact": E_INF, "systems": {}}
    print(f"Exact thermodynamic limit  E/L = 1 - 4 ln2 = {E_INF:.10f}\n", flush=True)

    for L in LS:
        print(f"=== L = {L} " + "=" * 40, flush=True)
        t0 = time.time()
        H = xxx_hamiltonian(L, J=1.0)
        sec = sz_sector_indices(L, L // 2)
        w_sec, _ = ground_state(H, k=1, sector=sec)
        E0 = float(w_sec[0])
        rec = {"E0": E0, "E0_per_site": E0 / L, "dim": 2**L,
               "dim_sz0": int(sec.size), "t_ed": time.time() - t0, "vqe": {}}
        # cross-check in the full space for the sizes where it is cheap
        if L <= 12:
            w_full, _ = ground_state(H, k=1)
            rec["E0_fullspace"] = float(w_full[0])
            rec["sector_consistency"] = abs(float(w_full[0]) - E0)
        print(f"  ED: E0 = {E0:+.10f}   E0/L = {E0/L:+.8f}   "
              f"dim(Sz=0) = {sec.size}   ({rec['t_ed']:.2f}s)", flush=True)

        for P in LAYER_PLAN[L]:
            ans = HVA(L, P, *xxx_bonds(L))
            t1 = time.time()
            print(f"    HVA P={P}  ({ans.n_params} params)", flush=True)
            res = vqe_adjoint(H, ans, RESTARTS[L], MAXITER[L], seed=1234 + L)
            err = res.energy - E0
            rec["vqe"][str(P)] = {
                "n_params": ans.n_params, "E": res.energy,
                "abs_err": err, "rel_err": abs(err / E0),
                "err_per_site": err / L, "t": time.time() - t1,
                "n_grad_calls": res.n_evals,
            }
            print(f"      -> E = {res.energy:+.10f}   dE = {err:.3e}   "
                  f"rel = {abs(err/E0):.3e}   ({time.time()-t1:.1f}s)", flush=True)
        out["systems"][str(L)] = rec
        with open("results/model1.json", "w") as f:
            json.dump(out, f, indent=2)

    # finite-size extrapolation of the ED energies
    Ls = np.array([L for L in LS], dtype=float)
    e = np.array([out["systems"][str(int(L))]["E0_per_site"] for L in Ls])
    A = np.vstack([np.ones_like(Ls), 1.0 / Ls**2]).T
    coef, *_ = np.linalg.lstsq(A, e, rcond=None)
    out["extrapolation"] = {"E_inf_fit": float(coef[0]), "slope_1_over_L2": float(coef[1]),
                            "deviation_from_bethe": float(coef[0] - E_INF)}
    print(f"\n  1/L^2 fit of E0/L  ->  {coef[0]:+.8f}   "
          f"(Bethe {E_INF:+.8f}, deviation {coef[0]-E_INF:+.2e})", flush=True)
    with open("results/model1.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
