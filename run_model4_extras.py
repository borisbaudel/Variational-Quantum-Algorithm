"""
run_model4_extras.py -- the two quantities that are usually asserted in a VQE
write-up and are measured here instead: the sampling cost, and whether a
Gaussian warm start actually helps.

A. Shot noise.  All energies elsewhere are exact expectation values.  Here the
   actual sampling cost is measured for the three models: the coefficient
   1-norm, the number of qubit-wise-commuting groups, the worst-case analytic
   budget, and the *empirical* variance of the grouped estimator, from which
   the shots needed for a target precision follow directly.

B. Gaussian (q = 2) warm start for SYK.  The free-fermion Majorana model
   H_2 = (i/2) sum K_ij chi_i chi_j has a Gaussian ground state, preparable on
   hardware in O(n^2) Givens rotations.  Two uses are tested:
     - as the ADAPT reference state, against the product-state references;
     - as the s = 0 end of an interpolation H(s) = (1-s) H_2 + s H_SYK, with
       the complex brickwork re-optimised along the path, against a cold start
       at s = 1.

Produces results/model4_extras.json
"""

import json
import math
import time

import numpy as np
import scipy.optimize as opt

from run_model3b import HardwareEfficientComplex, adapt_vqe, rz
from spinvqe.analysis import (q2_ground_state, qwc_groups, sample_energy_qwc,
                              shots_for_precision)
from spinvqe.ansatz import apply_1q, ry
from spinvqe.core import (ground_state, hs_hamiltonian, pauli_string,
                          sz_sector_indices, xxx_hamiltonian)
from spinvqe.qiskit_interface import hs_sparse_pauli, xxx_sparse_pauli
from spinvqe.syk import syk_hamiltonian
from spinvqe.vqe import run_vqe

EPS = 1e-3


def terms_from_sparse_pauli(op):
    """SparsePauliOp -> {string (site 0 first): coefficient}, undoing Qiskit's
    little-endian labelling so the keys match the rest of the codebase."""
    out = {}
    for p, c in zip(op.paulis, op.coeffs):
        s = str(p)[::-1]
        out[s] = out.get(s, 0j) + complex(c)
    return {s: c for s, c in out.items() if abs(c) > 1e-15}


def shot_study(name, terms, psi, rng, shots=2000, repeats=60):
    l1 = sum(abs(c) for c in terms.values())
    groups = qwc_groups([s for s in terms])
    exact = None
    ests = [sample_energy_qwc(psi, terms, shots, rng) for _ in range(repeats)]
    sd = float(np.std(ests))
    # Var scales as 1/shots_per_group, so shots for precision EPS:
    need = shots * (sd / EPS) ** 2
    rec = {"n_terms": len(terms), "l1_norm": float(l1),
           "n_qwc_groups": len(groups),
           "analytic_worst_case_shots": shots_for_precision(terms, EPS),
           "shots_per_group_used": shots, "empirical_sd": sd,
           "empirical_shots_per_group_for_1e-3": float(need),
           "empirical_total_shots_for_1e-3": float(need * len(groups))}
    print(f"  {name:<22s} {len(terms):5d} terms  ||c||_1 = {l1:8.3f}  "
          f"{len(groups):5d} QWC groups", flush=True)
    print(f"  {'':22s} sd({shots} shots/group) = {sd:.5f}  ->  "
          f"{need*len(groups):.3e} total shots for eps = 1e-3", flush=True)
    return rec


def main():
    out = {}
    rng = np.random.default_rng(7)

    # ----------------------------------------------------------- A. shots
    print("\nA. shot-noise cost of one energy evaluation (eps = 1e-3)", flush=True)
    A = {}
    L = 8
    Hx = xxx_hamiltonian(L)
    _, vx = ground_state(Hx, k=1)
    A["XXX_L8"] = shot_study("XXX  L=8", terms_from_sparse_pauli(xxx_sparse_pauli(L)),
                             vx[:, 0], rng)
    Hh = hs_hamiltonian(L)
    _, vh = ground_state(Hh, k=1)
    A["HS_L8"] = shot_study("Haldane-Shastry L=8",
                            terms_from_sparse_pauli(hs_sparse_pauli(L)),
                            vh[:, 0], rng)
    for N in (8, 12):
        Hs, _, agg = syk_hamiltonian(N, seed=100)
        w, v = np.linalg.eigh(Hs.toarray())
        A[f"SYK_N{N}"] = shot_study(f"SYK  N={N}", agg, v[:, 0], rng)
    out["shot_noise"] = A
    json.dump(out, open("results/model4_extras.json", "w"), indent=2)

    # ------------------------------------------------ B. Gaussian warm start
    print("\nB. q = 2 Gaussian warm start for SYK", flush=True)
    B = {}
    for N in (10, 12, 14):
        n = N // 2
        H, _, agg = syk_hamiltonian(N, seed=100)
        w, v = np.linalg.eigh(H.toarray())
        E0 = float(w[0])
        H2, E2, psi2 = q2_ground_state(N, seed=2)
        ov = float(abs(np.vdot(v[:, 0], psi2)))
        print(f"  N={N}: E0(SYK) = {E0:+.6f}   E0(q=2) = {E2:+.6f}   "
              f"|<psi_SYK|psi_q2>| = {ov:.4f}", flush=True)
        rec = {"E0": E0, "E0_q2": E2, "overlap_q2_with_syk_gs": ov,
               "energy_of_q2_state_under_H_syk":
                   float(np.real(np.vdot(psi2, H @ psi2)))}

        # --- B1: ADAPT from the Gaussian reference
        pool = [pauli_string(s) for s in
                sorted(agg, key=lambda s: -abs(agg[s]))[:min(300, len(agg))]]
        t0 = time.time()
        E_g, p_g, hist = adapt_vqe(H, pool, psi2.astype(complex),
                                   max_ops=min(60, 6 * n), verbose=False)
        rec["adapt_from_q2"] = {"n_ops": int(len(p_g)), "E": E_g,
                                "rel_err": abs((E_g - E0) / E0),
                                "t": time.time() - t0}
        print(f"      ADAPT from q=2 state : E = {E_g:+.8f}  {len(p_g)} ops  "
              f"rel err = {100*abs((E_g-E0)/E0):6.2f}%", flush=True)

        # --- B2: adiabatic path with the complex brickwork
        ans = HardwareEfficientComplex(n, 4)
        rng2 = np.random.default_rng(3)
        params = rng2.uniform(-0.2, 0.2, ans.n_params)
        path = []
        for s in (0.0, 0.25, 0.5, 0.75, 1.0):
            Hs_mix = (1 - s) * H2 + s * H

            def f(p):
                psi = ans.state(p)
                return float(np.real(np.vdot(psi, Hs_mix @ psi)))

            r = opt.minimize(f, params, method="L-BFGS-B",
                             options={"maxiter": 300, "ftol": 1e-13})
            params = r.x
            path.append({"s": s, "E": float(r.fun)})
        E_path = path[-1]["E"]
        rec["adiabatic_path"] = {"path": path, "E": E_path,
                                 "rel_err": abs((E_path - E0) / E0)}
        print(f"      brickwork, adiabatic : E = {E_path:+.8f}  "
              f"rel err = {100*abs((E_path-E0)/E0):6.2f}%", flush=True)

        # --- B3: the same ansatz, cold start at s = 1 (the control)
        r_cold = run_vqe(H, ans, n_restarts=3, seed=3, maxiter=500)
        rec["cold_start"] = {"E": r_cold.energy,
                             "rel_err": abs((r_cold.energy - E0) / E0)}
        print(f"      brickwork, cold start: E = {r_cold.energy:+.8f}  "
              f"rel err = {100*abs((r_cold.energy-E0)/E0):6.2f}%\n", flush=True)

        B[str(N)] = rec
        out["gaussian_warm_start"] = B
        json.dump(out, open("results/model4_extras.json", "w"), indent=2)

    print("written results/model4_extras.json", flush=True)


if __name__ == "__main__":
    main()
