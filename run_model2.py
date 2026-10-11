"""
run_model2.py -- Model 2: the Haldane-Shastry chain.

Three things are established here, in order:

1. the geometric identity  |z_j - z_k|^2 = 4 sin^2( pi (j-k) / L ),  checked
   numerically operator-by-operator, which turns the Hamiltonian of the
   assignment into the textbook Haldane-Shastry chain;

2. the closed-form ground-state energy  E_0 = -(pi^2 J / 24)(L + 5/L),
   checked against exact diagonalisation at every accessible L -- so it is
   used as a benchmark only after being verified, not on authority;

3. the exact ground state itself: the Gutzwiller-Jastrow wavefunction
       psi(n_1..n_M) = prod_{i<j} (z_{n_i} - z_{n_j})^2   (M = L/2 down spins)
   is constructed explicitly in the S^z = 0 sector and verified to be an
   eigenvector of H.  This is the *problem-specific ansatz* for Model 2: it
   has zero variational parameters and is exact, which is the honest answer to
   "which ansatz would you choose".

Produces results/model2.json
"""

import itertools
import json
import time

import numpy as np

from spinvqe.core import (ground_state, hs_coupling, hs_exact_energy,
                          hs_hamiltonian, sz_sector_indices)

LS = [4, 6, 8, 10, 12, 14]


def check_identity(L: int) -> float:
    """max |4 sin^2(pi(j-k)/L) - |z_j - z_k|^2| over all pairs."""
    err = 0.0
    for j in range(L):
        for k in range(j + 1, L):
            zj, zk = np.exp(2j * np.pi * j / L), np.exp(2j * np.pi * k / L)
            err = max(err, abs(4 * np.sin(np.pi * (j - k) / L) ** 2 - abs(zj - zk) ** 2))
    return err


def jastrow_ground_state(L: int, extra_phase: bool = False):
    """Gutzwiller-Jastrow state in the S^z = 0 sector.

    Basis convention of core.py: bit 0 == spin up, bit 1 == spin down, site 0
    is the most significant bit.  The amplitude of a configuration with down
    spins at sites n_1 < ... < n_M is prod_{i<j}(z_{n_i} - z_{n_j})^2, times
    prod_i z_{n_i} if `extra_phase` (some conventions include this momentum
    factor; both are tested).
    """
    M = L // 2
    z = np.exp(2j * np.pi * np.arange(L) / L)
    sec = sz_sector_indices(L, L - M)  # n_up = L - M
    amp = np.zeros(sec.size, dtype=complex)
    for a, idx in enumerate(sec):
        downs = [s for s in range(L) if (idx >> (L - 1 - s)) & 1]
        val = 1.0 + 0j
        for i in range(len(downs)):
            for j in range(i + 1, len(downs)):
                val *= (z[downs[i]] - z[downs[j]]) ** 2
            if extra_phase:
                val *= z[downs[i]]
        amp[a] = val
    nrm = np.linalg.norm(amp)
    return amp / nrm, sec


def main():
    out = {"systems": {}}
    print("Model 2 -- Haldane-Shastry\n", flush=True)

    for L in LS:
        t0 = time.time()
        id_err = check_identity(L)
        H_id = hs_hamiltonian(L, use_identity=True)
        H_lit = hs_hamiltonian(L, use_identity=False)
        op_err = float(abs(H_id - H_lit).max())
        herm = float(abs(H_id - H_id.getH()).max())

        sec = sz_sector_indices(L, L // 2)
        E0 = float(ground_state(H_id, k=1, sector=sec)[0][0])
        E_formula = hs_exact_energy(L)

        rec = {
            "identity_max_err": float(id_err),
            "operator_max_err": op_err,
            "hermiticity_err": herm,
            "E0_ED": E0,
            "E0_formula": E_formula,
            "formula_abs_err": abs(E0 - E_formula),
            "formula_rel_err": abs((E0 - E_formula) / E0),
            "E0_per_site": E0 / L,
            "dim_sz0": int(sec.size),
            "t": time.time() - t0,
        }

        # --- exact Jastrow ground state
        Hs = H_id[sec][:, sec]
        for phase in (False, True):
            psi, _ = jastrow_ground_state(L, extra_phase=phase)
            Hpsi = Hs @ psi
            e = complex(np.vdot(psi, Hpsi)).real
            resid = float(np.linalg.norm(Hpsi - e * psi))
            key = "jastrow_with_phase" if phase else "jastrow"
            rec[key] = {"energy": e, "abs_err_vs_ED": abs(e - E0),
                        "eigen_residual": resid,
                        "overlap_with_gs": None}
        # overlap with the ED ground state, for the better of the two
        _, v = ground_state(H_id, k=1, sector=sec)
        gs = v[:, 0]
        for phase, key in ((False, "jastrow"), (True, "jastrow_with_phase")):
            psi, _ = jastrow_ground_state(L, extra_phase=phase)
            rec[key]["overlap_with_gs"] = float(abs(np.vdot(gs, psi)))

        out["systems"][str(L)] = rec
        print(f"L={L:3d}  E0(ED) = {E0:+.10f}   formula = {E_formula:+.10f}   "
              f"|diff| = {abs(E0-E_formula):.2e}", flush=True)
        print(f"        identity err {id_err:.1e}   operator err {op_err:.1e}   "
              f"dim(Sz=0) {sec.size}", flush=True)
        print(f"        Jastrow: E = {rec['jastrow']['energy']:+.10f}  "
              f"residual = {rec['jastrow']['eigen_residual']:.2e}  "
              f"|<gs|psi>| = {rec['jastrow']['overlap_with_gs']:.10f}", flush=True)
        print(f"        Jastrow x prod z: residual = "
              f"{rec['jastrow_with_phase']['eigen_residual']:.2e}  "
              f"|<gs|psi>| = {rec['jastrow_with_phase']['overlap_with_gs']:.10f}\n",
              flush=True)

        with open("results/model2.json", "w") as f:
            json.dump(out, f, indent=2)

    # thermodynamic limit: E0/L -> -pi^2/24
    print(f"  -pi^2/24 = {-np.pi**2/24:.10f}   "
          f"(E0/L at L={LS[-1]} is {out['systems'][str(LS[-1])]['E0_per_site']:+.10f})",
          flush=True)
    out["E_inf_exact"] = -np.pi**2 / 24
    with open("results/model2.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
