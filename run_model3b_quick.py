"""
run_model3b_quick.py -- the same comparison as run_model3b.py, with the
brickwork depth sweep trimmed to P = 4 (already exact at the smallest sizes)
so that the ADAPT results are obtained for N = 10, 12, 14 in minutes rather
than hours.  Results are merged into results/model3b.json; entries already
present from the full run are kept untouched.
"""

import json
import os
import time

import numpy as np

from run_model3b import (HardwareEfficientComplex, adapt_vqe,
                         complexity_of_ground_state, rz)
from spinvqe.ansatz import HardwareEfficient, apply_1q, ry
from spinvqe.core import pauli_string
from spinvqe.syk import syk_hamiltonian
from spinvqe.vqe import run_vqe

OUT = "results/model3b.json"


def main():
    out = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for N in (10, 12, 14):
        if str(N) in out and "adapt" in out[str(N)]:
            print(f"N={N} already complete, skipping", flush=True)
            continue
        n = N // 2
        H, terms, agg = syk_hamiltonian(N, seed=100)
        w, v = np.linalg.eigh(H.toarray())
        E0, deg = float(w[0]), int(np.sum(w - w[0] < 1e-9))
        cplx = complexity_of_ground_state(v[:, 0])
        print(f"=== N={N} (n={n})  E0={E0:+.8f}  deg={deg}  "
              f"imag weight={cplx:.4f}", flush=True)
        rec = {"E0": E0, "degeneracy": deg, "gs_imaginary_weight": cplx,
               "n_terms": len(agg), "real_brickwork": {}, "complex_brickwork": {}}

        for tag, cls, store in (("RY+CZ", HardwareEfficient, "real_brickwork"),
                                ("RY+RZ+CZ", HardwareEfficientComplex,
                                 "complex_brickwork")):
            ans = cls(n, 4)
            r = run_vqe(H, ans, n_restarts=3, seed=3, maxiter=500)
            rec[store]["4"] = {"n_params": ans.n_params, "E": r.energy,
                               "rel_err": abs((r.energy - E0) / E0)}
            print(f"    {tag:10s} P= 4 ({ans.n_params:3d} params): "
                  f"E={r.energy:+.6f}   rel err = "
                  f"{100*abs((r.energy-E0)/E0):5.1f}%", flush=True)

        pool = [pauli_string(s) for s in
                sorted(agg, key=lambda s: -abs(agg[s]))[:min(300, len(agg))]]

        def product_ref(complex_phases):
            r = np.zeros(2**n, dtype=complex)
            r[0] = 1.0
            for q in range(n):
                r = apply_1q(r, ry(0.7 + 0.13 * q), q, n)
                if complex_phases:
                    r = apply_1q(r, rz(0.4 + 0.21 * q), q, n)
            return r

        rec["adapt"] = {}
        for tag, cp in (("real_reference", False), ("complex_reference", True)):
            t0 = time.time()
            E_ad, p_ad, hist = adapt_vqe(H, pool, product_ref(cp),
                                         max_ops=min(60, 6 * n), verbose=False)
            rec["adapt"][tag] = {
                "pool_size": len(pool), "n_ops": int(len(p_ad)), "E": E_ad,
                "rel_err": abs((E_ad - E0) / E0), "history": hist,
                "t": time.time() - t0}
            print(f"    ADAPT ({tag:17s}): E={E_ad:+.8f}  {len(p_ad):2d} ops"
                  f"   rel err = {100*abs((E_ad-E0)/E0):6.2f}%  "
                  f"({time.time()-t0:.1f}s)", flush=True)

        out[str(N)] = rec
        json.dump(out, open(OUT, "w"), indent=2)
    print("done", flush=True)


if __name__ == "__main__":
    main()
