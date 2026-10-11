"""
run_model1_warmstart.py -- Model 1, second strategy: *layerwise growth*
(adiabatic warm start).

Motivation.  With random initialisation the HVA converges to the exact ground
state for L <= 8, but from L = 12 onwards the optimiser stalls near a relative
error of a few percent and adding layers does not help (see results/model1.json).
That is a classic HVA local minimum, not a limit of the ansatz.

Strategy.  Optimise P = 1.  Then initialise the (P+1)-layer circuit with the
converged P-layer parameters and a nearly-identity new layer
(theta_new ~ 0), and re-optimise.  Each step starts in a region that already
represents the ground state to the accuracy of the shorter circuit, so the
optimiser only has to improve it -- the discrete analogue of adiabatic state
preparation.

Produces results/model1_warmstart.json
"""

import json
import time

import numpy as np
import scipy.optimize as opt

from spinvqe.ansatz import HVA
from spinvqe.core import ground_state, sz_sector_indices, xxx_bonds, xxx_hamiltonian

E_INF = 1.0 - 4.0 * np.log(2.0)
LS = [4, 6, 8, 12, 16]
PMAX = {4: 4, 6: 4, 8: 6, 12: 8, 16: 10}


def optimise(H, ansatz, x0, maxiter=1500):
    res = opt.minimize(lambda p: ansatz.energy_and_grad(p, H), x0, jac=True,
                       method="L-BFGS-B",
                       options={"maxiter": maxiter, "ftol": 1e-15, "gtol": 1e-12})
    return float(res.fun), res.x, int(res.nit)


def main():
    out = {"E_inf_exact": E_INF, "systems": {}}
    for L in LS:
        H = xxx_hamiltonian(L)
        sec = sz_sector_indices(L, L // 2)
        E0 = float(ground_state(H, k=1, sector=sec)[0][0])
        print(f"=== L={L}   E0(ED) = {E0:+.10f}", flush=True)
        rec = {"E0": E0, "ladder": {}}

        rng = np.random.default_rng(20240 + L)
        # --- P = 1 : a couple of restarts, it is a two-parameter problem
        best = None
        ans1 = HVA(L, 1, *xxx_bonds(L))
        for _ in range(12):
            E, x, _ = optimise(H, ans1, rng.uniform(-np.pi / 2, np.pi / 2, 2))
            if best is None or E < best[0]:
                best = (E, x)
        E, x = best
        rec["ladder"]["1"] = {"E": E, "abs_err": E - E0, "rel_err": abs((E - E0) / E0),
                              "n_params": 2}
        print(f"    P=1  E={E:+.10f}  dE={E-E0:.3e}", flush=True)

        # --- grow one layer at a time
        for P in range(2, PMAX[L] + 1):
            ans = HVA(L, P, *xxx_bonds(L))
            t0 = time.time()
            cands = []
            for eps in (0.0, 0.05, -0.05, 0.2):
                x0 = np.concatenate([x, [eps, eps]])
                cands.append(optimise(H, ans, x0))
            # one fully random restart as a control
            cands.append(optimise(H, ans, rng.uniform(-0.5, 0.5, ans.n_params)))
            Eb, xb, _ = min(cands, key=lambda c: c[0])
            E, x = Eb, xb
            rec["ladder"][str(P)] = {"E": E, "abs_err": E - E0,
                                     "rel_err": abs((E - E0) / E0),
                                     "n_params": ans.n_params, "t": time.time() - t0}
            print(f"    P={P:<2d} E={E:+.10f}  dE={E-E0:.3e}  rel={abs((E-E0)/E0):.3e}"
                  f"  ({time.time()-t0:.1f}s)", flush=True)
        out["systems"][str(L)] = rec
        with open("results/model1_warmstart.json", "w") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
