"""make_figures.py -- all plots for the report, from the JSON result files."""

import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9.5,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "figure.dpi": 160, "savefig.bbox": "tight", "axes.grid": True,
    "grid.alpha": 0.25, "grid.linewidth": 0.5, "lines.linewidth": 1.4,
    "lines.markersize": 4.5, "axes.spines.top": False, "axes.spines.right": False,
})
NAVY, RED, GOLD, TEAL, GREY = "#0F2440", "#B03A2E", "#C8A24A", "#1F6F6B", "#8A9099"
COL = [NAVY, RED, TEAL, GOLD, "#6A3D9A"]
FIG = "figures"


def load(p):
    with open(p) as f:
        return json.load(f)


# ---------------------------------------------------------------- Model 1
def fig_model1():
    r = load("results/model1.json")
    w = load("results/model1_warmstart.json")
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9))

    for c, L in zip(COL, sorted(map(int, r["systems"]))):
        v = r["systems"][str(L)]["vqe"]
        Ps = sorted(map(int, v))
        ax[0].semilogy(Ps, [max(abs(v[str(p)]["rel_err"]), 1e-17) for p in Ps],
                       "o--", color=c, label=f"$L={L}$", alpha=0.85)
        if str(L) not in w["systems"]:
            continue
        lad = w["systems"][str(L)]["ladder"]
        Pw = sorted(map(int, lad))
        ax[1].semilogy(Pw, [max(abs(lad[str(p)]["rel_err"]), 1e-17) for p in Pw],
                       "s-", color=c, label=f"$L={L}$")
    for a, t in zip(ax, ["random initialisation", "layerwise warm start"]):
        a.set_xlabel("HVA layers $P$")
        a.set_title(t)
        a.axhline(1e-3, color=GREY, lw=0.7, ls=":")
    ax[0].set_ylabel(r"relative error $|E_{\rm VQE}-E_0|/|E_0|$")
    ax[0].set_ylim(1e-17, 1)
    ax[1].set_ylim(1e-17, 1)
    ax[1].legend(ncol=2, frameon=False)
    fig.savefig(f"{FIG}/model1_error_vs_layers.pdf")
    plt.close(fig)

    # finite-size scaling
    fig, a = plt.subplots(figsize=(3.5, 2.7))
    Ls = np.array(sorted(map(int, r["systems"])), dtype=float)
    e = np.array([r["systems"][str(int(L))]["E0_per_site"] for L in Ls])
    x = 1.0 / Ls**2
    a.plot(x, e, "o", color=NAVY, label="exact diagonalisation")
    A = np.vstack([np.ones_like(x), x]).T
    coef, *_ = np.linalg.lstsq(A, e, rcond=None)
    xx = np.linspace(0, x.max() * 1.05, 50)
    a.plot(xx, coef[0] + coef[1] * xx, "-", color=RED, lw=1.1,
           label=rf"fit $\to {coef[0]:.5f}$")
    a.axhline(r["E_inf_exact"], color=GOLD, ls="--", lw=1.1,
              label=rf"Bethe $1-4\ln 2={r['E_inf_exact']:.5f}$")
    a.set_xlabel(r"$1/L^2$")
    a.set_ylabel(r"$E_0/L$")
    a.legend(frameon=False, loc="lower right")
    fig.savefig(f"{FIG}/model1_finite_size.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- Model 2
def fig_model2():
    r = load("results/model2.json")
    Ls = np.array(sorted(map(int, r["systems"])), dtype=float)
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9))

    err = [r["systems"][str(int(L))]["formula_abs_err"] for L in Ls]
    res = [r["systems"][str(int(L))]["jastrow_with_phase"]["eigen_residual"] for L in Ls]
    idn = [r["systems"][str(int(L))]["identity_max_err"] for L in Ls]
    ax[0].semilogy(Ls, np.maximum(err, 1e-18), "o-", color=NAVY,
                   label=r"$|E_0^{\rm ED}-E_0^{\rm formula}|$")
    ax[0].semilogy(Ls, np.maximum(res, 1e-18), "s-", color=RED,
                   label=r"$\|H\psi_{\rm Jastrow}-E\psi_{\rm Jastrow}\|$")
    ax[0].semilogy(Ls, np.maximum(idn, 1e-18), "^-", color=TEAL,
                   label=r"$\max||z_j-z_k|^2-4\sin^2|$")
    ax[0].axhspan(1e-18, 1e-13, color=GOLD, alpha=0.18)
    ax[0].text(Ls[0], 2e-16, "machine precision", fontsize=7, color="#7a6520")
    ax[0].set_xlabel("$L$")
    ax[0].set_ylabel("absolute deviation")
    ax[0].set_ylim(1e-18, 1e-10)
    ax[0].legend(frameon=False, loc="upper left")
    ax[0].set_title("everything analytic is verified, not assumed")

    e = np.array([r["systems"][str(int(L))]["E0_per_site"] for L in Ls])
    ax[1].plot(1 / Ls**2, e, "o", color=NAVY, label="ED")
    xx = np.linspace(0, (1 / Ls**2).max() * 1.05, 50)
    ax[1].plot(xx, -np.pi**2 / 24 - np.pi**2 / 24 * 5 * xx, "-", color=RED, lw=1.1,
               label=r"$-\frac{\pi^2}{24}\left(1+\frac{5}{L^2}\right)$")
    ax[1].axhline(-np.pi**2 / 24, color=GOLD, ls="--", lw=1.1,
                  label=r"$-\pi^2/24$")
    ax[1].set_xlabel(r"$1/L^2$")
    ax[1].set_ylabel(r"$E_0/L$")
    ax[1].legend(frameon=False, loc="lower left")
    ax[1].set_title("Haldane-Shastry, exact at every $L$")
    fig.savefig(f"{FIG}/model2_verification.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- Model 3
def fig_model3():
    r = load("results/model3.json")
    fig, ax = plt.subplots(1, 3, figsize=(10.6, 2.9))

    # (a) term count and grouping
    c = r["counting"]
    Ns = np.array(sorted(map(int, c)), dtype=int)
    nt = [c[str(N)]["n_terms"] for N in Ns]
    ax[0].semilogy(Ns, nt, "o-", color=NAVY, label=r"Pauli strings $\binom{N}{4}$")
    Ng = [N for N in Ns if "n_groups_commuting" in c[str(N)]]
    ax[0].semilogy(Ng, [c[str(N)]["n_groups_commuting"] for N in Ng], "s-",
                   color=RED, label="commuting groups")
    ax[0].semilogy(Ng, [c[str(N)]["n_groups_qwc"] for N in Ng], "^-",
                   color=TEAL, label="qubit-wise-commuting groups")
    ax[0].set_xlabel("Majoranas $N$")
    ax[0].set_ylabel("count")
    ax[0].legend(frameon=False, loc="upper left")
    ax[0].set_title("(a) measurement cost")

    # (b) entanglement: volume vs log
    ed = r["exact_diagonalisation"]
    Ns2 = np.array(sorted(map(int, ed)), dtype=int)
    nq = np.array([ed[str(N)]["n_qubits"] for N in Ns2])
    S = np.array([ed[str(N)]["S_half_mean"] for N in Ns2])
    Se = np.array([ed[str(N)]["S_half_std"] for N in Ns2])
    ax[1].errorbar(nq, S, yerr=Se, fmt="o-", color=RED, capsize=2,
                   label="SYK ground state")
    xe = r["xxx_entanglement"]
    Lx = np.array(sorted(map(int, xe)), dtype=int)
    ax[1].plot(Lx, [xe[str(L)]["S_half"] for L in Lx], "s-", color=NAVY,
               label="XXX ground state")
    nn = np.linspace(nq.min(), nq.max(), 40)
    ax[1].plot(nn, (nn // 2) * np.log(2), "--", color=GOLD, lw=1.1,
               label=r"maximal $\frac{n}{2}\ln 2$")
    ax[1].set_xlabel("qubits $n$")
    ax[1].set_ylabel("half-cut entropy (nats)")
    ax[1].legend(frameon=False, loc="upper left")
    ax[1].set_title("(b) volume law vs logarithm")

    # (c) barren plateau
    bp = r["barren_plateau"]
    Ns3 = np.array(sorted(map(int, bp)), dtype=int)
    nq3 = [bp[str(N)]["n_qubits"] for N in Ns3]
    var = [bp[str(N)]["grad_var"] for N in Ns3]
    ax[2].semilogy(nq3, var, "o-", color=NAVY, label=r"Var$[\partial_\theta E]$")
    co = np.polyfit(nq3, np.log(var), 1)
    ax[2].semilogy(nq3, np.exp(np.polyval(co, nq3)), "--", color=RED, lw=1.1,
                   label="guide to the eye")
    ax[2].set_xlabel("qubits $n = N/2$")
    ax[2].set_ylabel(r"Var$[\partial E/\partial\theta_0]$")
    ax[2].legend(frameon=False)
    ax[2].set_title("(c) flat gradients, brickwork ansatz")
    fig.savefig(f"{FIG}/model3_obstructions.pdf")
    plt.close(fig)

    # naive VQE failure
    if "naive_vqe" in r:
        fig, a = plt.subplots(figsize=(3.6, 2.7))
        for col, N in zip(COL, sorted(map(int, r["naive_vqe"]))):
            sw = r["naive_vqe"][str(N)]["sweep"]
            Ps = sorted(map(int, sw))
            a.semilogy([sw[str(p)]["n_params"] for p in Ps],
                       [abs(sw[str(p)]["rel_err"]) for p in Ps],
                       "o-", color=col, label=f"$N={N}$")
        a.set_xlabel("number of parameters")
        a.set_ylabel(r"relative error in $E_0$")
        a.axhline(1e-2, color=GREY, ls=":", lw=0.8)
        a.legend(frameon=False)
        a.set_title("naive VQE on SYK: more parameters do not help")
        fig.savefig(f"{FIG}/model3_naive_vqe.pdf")
        plt.close(fig)




def fig_model4():
    """SYK: every strategy tried, on one axis."""
    b = load("results/model3b.json")
    w = load("results/model4_extras.json")["gaussian_warm_start"]
    Ns = sorted(map(int, b))
    fig, ax = plt.subplots(figsize=(4.6, 3.1))
    series = [
        ("RY+CZ (real)", NAVY, "o--",
         [100 * b[str(N)]["real_brickwork"][max(b[str(N)]["real_brickwork"], key=int)]["rel_err"] for N in Ns]),
        ("RY+RZ+CZ (complex)", RED, "s-",
         [100 * b[str(N)]["complex_brickwork"][max(b[str(N)]["complex_brickwork"], key=int)]["rel_err"] for N in Ns]),
        ("ADAPT, product ref.", TEAL, "^-",
         [100 * b[str(N)]["adapt"]["complex_reference"]["rel_err"] for N in Ns]),
        ("ADAPT, Gaussian $q{=}2$ ref.", GOLD, "D-",
         [100 * w[str(N)]["adapt_from_q2"]["rel_err"] if str(N) in w else np.nan for N in Ns]),
    ]
    for lab, col, mk, y in series:
        y = np.array(y, dtype=float)
        y[y <= 0] = 1e-3
        ax.semilogy(Ns, y, mk, color=col, label=lab)
    ax.set_xlabel("Majoranas $N$")
    ax.set_ylabel(r"relative error in $E_0$ (percent)")
    ax.set_xticks(Ns)
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    ax.set_title("SYK: the whole ladder of strategies")
    fig.savefig(f"{FIG}/model4_strategies.pdf")
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    done = []
    for name, fn in (("model1", fig_model1), ("model2", fig_model2),
                     ("model3", fig_model3), ("model4", fig_model4)):
        try:
            fn()
            done.append(name)
        except Exception as exc:  # a run may still be in flight
            print(f"  skipped {name}: {exc}")
    print("figures written for:", ", ".join(done))
