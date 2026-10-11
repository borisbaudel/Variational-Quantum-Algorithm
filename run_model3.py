"""
run_model3.py -- Model 3: the SYK model.

The assignment warns that "a naive implementation will probably fail".  This
script establishes *why*, quantitatively, and then measures which of the
standard remedies actually buys anything.  Four separate obstructions are
measured, because they have different cures:

  (a) term count.  H is a sum of exactly C(N,4) Pauli strings on N/2 qubits.
      Measured, together with the number of simultaneously measurable groups
      obtained from a greedy colouring of the non-commutation graph.
  (b) operator weight.  The Jordan-Wigner Z-strings make the typical Pauli
      string act on O(N/2) qubits, so the terms are global, not local.
  (c) entanglement.  The half-cut entropy of the SYK ground state is compared
      with that of the XXX ground state: volume law versus logarithm.  This is
      the obstruction that no amount of optimiser tuning can fix, because it
      is a statement about the *expressivity* of a shallow circuit.
  (d) barren plateaus.  Variance of dE/dtheta at random initialisation, as a
      function of qubit number, for the hardware-efficient ansatz.

Produces results/model3.json
"""

import itertools
import json
import math
import time

import numpy as np

from spinvqe.ansatz import HardwareEfficient
from spinvqe.core import ground_state, xxx_hamiltonian, sz_sector_indices
from spinvqe.syk import (check_clifford_algebra, group_strings, syk_couplings,
                         syk_hamiltonian, syk_pauli_terms)
from spinvqe.vqe import gradient_variance, run_vqe


def half_cut_entropy(psi: np.ndarray, n: int) -> float:
    """von Neumann entropy (in nats) of the left n//2 qubits."""
    nl = n // 2
    m = psi.reshape(2**nl, 2 ** (n - nl))
    s = np.linalg.svd(m, compute_uv=False)
    p = s**2
    p = p[p > 1e-14]
    p = p / p.sum()
    return float(-(p * np.log(p)).sum())


def main():
    out = {}

    # ----- (0) the mapping is correct
    print("--- Jordan-Wigner Majorana algebra", flush=True)
    algebra = {}
    for N in (4, 6, 8, 10, 12):
        ok = check_clifford_algebra(N)
        algebra[str(N)] = bool(ok)
        print(f"    N={N:2d}: {{chi_a, chi_b}} = 2 delta_ab  ->  {ok}", flush=True)
    out["clifford_algebra_verified"] = algebra

    # ----- (a) term counting and measurement grouping
    print("\n--- term count and measurement grouping", flush=True)
    counting = {}
    for N in (8, 10, 12, 14, 16, 20, 24, 28, 32):
        n = N // 2
        n_terms = math.comb(N, 4)
        rec = {"n_qubits": n, "n_terms": n_terms, "hilbert_dim": 2**n}
        if N <= 16:
            H, terms, agg = syk_hamiltonian(N, seed=1)
            strings = [s for s in agg if abs(agg[s]) > 1e-15]
            rec["n_distinct_strings"] = len(strings)
            w = np.array([sum(c != "I" for c in s) for s in strings])
            rec["mean_weight"] = float(w.mean())
            rec["max_weight"] = int(w.max())
            t0 = time.time()
            rec["n_groups_commuting"] = group_strings(strings, "commute")
            rec["n_groups_qwc"] = group_strings(strings, "qwc")
            rec["t_grouping"] = time.time() - t0
            print(f"    N={N:2d} n={n}: {n_terms:6d} terms, {len(strings):6d} distinct, "
                  f"mean weight {w.mean():4.1f}/{n}, "
                  f"{rec['n_groups_commuting']:5d} commuting groups, "
                  f"{rec['n_groups_qwc']:5d} QWC groups", flush=True)
        else:
            print(f"    N={N:2d} n={n}: {n_terms:6d} terms  (grouping not attempted)",
                  flush=True)
        counting[str(N)] = rec
    out["counting"] = counting
    json.dump(out, open("results/model3.json", "w"), indent=2)

    # ----- exact spectra, disorder averaged
    print("\n--- exact diagonalisation, disorder averaged", flush=True)
    ed = {}
    for N in (8, 10, 12, 14, 16, 18, 20):
        n = N // 2
        E0s, ents, gaps, degs = [], [], [], []
        n_real = 20 if N <= 14 else (10 if N <= 18 else 5)
        t0 = time.time()
        for s in range(n_real):
            H, _, _ = syk_hamiltonian(N, J=1.0, seed=100 + s)
            k = min(8, 2**n - 1)
            if 2**n <= 1024:
                w, v = np.linalg.eigh(H.toarray())
            else:
                w, v = ground_state(H, k=k)
            E0s.append(w[0])
            deg = int(np.sum(w - w[0] < 1e-9))
            degs.append(deg)
            gaps.append(w[deg] - w[0] if deg < len(w) else np.nan)
            ents.append(half_cut_entropy(v[:, 0], n))
        ed[str(N)] = {
            "n_qubits": n, "n_realisations": n_real,
            "E0_mean": float(np.mean(E0s)), "E0_std": float(np.std(E0s)),
            "E0_per_majorana": float(np.mean(E0s) / N),
            "gap_above_gs_mean": float(np.nanmean(gaps)),
            "gs_degeneracy_mode": int(max(set(degs), key=degs.count)),
            "S_half_mean": float(np.mean(ents)), "S_half_std": float(np.std(ents)),
            "S_over_maximal": float(np.mean(ents) / ((n // 2) * np.log(2))),
            "t": time.time() - t0,
        }
        print(f"    N={N:2d} n={n}: E0/N = {np.mean(E0s)/N:+.6f} +- {np.std(E0s)/N:.6f}   "
              f"deg = {ed[str(N)]['gs_degeneracy_mode']}   "
              f"gap = {np.nanmean(gaps):.4f}   S_half = {np.mean(ents):.4f} nats "
              f"({ed[str(N)]['S_over_maximal']*100:.1f}% of maximal)", flush=True)
    out["exact_diagonalisation"] = ed
    json.dump(out, open("results/model3.json", "w"), indent=2)

    # ----- (c) comparison with the XXX chain: volume law vs logarithm
    print("\n--- half-cut entanglement: SYK vs XXX", flush=True)
    xxx_ent = {}
    for L in (6, 8, 10, 12, 14):
        H = xxx_hamiltonian(L)
        w, v = ground_state(H, k=1)
        psi = v[:, 0]
        S = half_cut_entropy(psi, L)
        xxx_ent[str(L)] = {"S_half": S, "S_over_maximal": S / ((L // 2) * np.log(2))}
        print(f"    XXX L={L:2d}: S_half = {S:.4f} nats "
              f"({S/((L//2)*np.log(2))*100:.1f}% of maximal)", flush=True)
    out["xxx_entanglement"] = xxx_ent
    json.dump(out, open("results/model3.json", "w"), indent=2)

    # ----- (d) barren plateaus
    print("\n--- gradient variance at random initialisation (brickwork ansatz)", flush=True)
    bp = {}
    for N in (8, 10, 12, 14, 16):
        n = N // 2
        H, _, _ = syk_hamiltonian(N, seed=7)
        var, absmean = gradient_variance(H, lambda: HardwareEfficient(n, 2 * n),
                                         n_samples=200, seed=11)
        bp[str(N)] = {"n_qubits": n, "grad_var": var, "grad_abs_mean": absmean}
        print(f"    N={N:2d} n={n}: Var[dE/dtheta_0] = {var:.3e}   "
              f"E[|dE/dtheta_0|] = {absmean:.3e}", flush=True)
    out["barren_plateau"] = bp
    json.dump(out, open("results/model3.json", "w"), indent=2)

    # ----- naive VQE, deliberately as the negative control
    print("\n--- naive VQE with the hardware-efficient ansatz", flush=True)
    naive = {}
    for N in (8, 10, 12, 16):
        n = N // 2
        H, _, _ = syk_hamiltonian(N, seed=100)
        E0 = float(np.linalg.eigvalsh(H.toarray())[0])
        row = {}
        for P in (2, 4, 8):
            ans = HardwareEfficient(n, P)
            t0 = time.time()
            res = run_vqe(H, ans, n_restarts=3, seed=5, maxiter=600)
            row[str(P)] = {"n_params": ans.n_params, "E": res.energy,
                           "abs_err": res.energy - E0,
                           "rel_err": abs((res.energy - E0) / E0),
                           "t": time.time() - t0}
            print(f"    N={N:2d} P={P:2d} ({ans.n_params:3d} params): "
                  f"E = {res.energy:+.6f}  vs E0 = {E0:+.6f}   "
                  f"rel err = {abs((res.energy-E0)/E0)*100:5.1f}%", flush=True)
        naive[str(N)] = {"E0": E0, "sweep": row}
        out["naive_vqe"] = naive
        with open("results/model3.json", "w") as f:
            json.dump(out, f, indent=2)
    print("\nwritten results/model3.json", flush=True)


if __name__ == "__main__":
    main()
