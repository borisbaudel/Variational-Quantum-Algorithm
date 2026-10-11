"""make_tables.py -- emit LaTeX table fragments from the JSON results, so that
every number in the report comes from a run and not from a keyboard."""

import json
import math
import os

os.makedirs("tables", exist_ok=True)


def load(p):
    with open(p) as f:
        return json.load(f)


def w(name, body):
    with open(f"tables/{name}.tex", "w") as f:
        f.write(body)
    print("wrote tables/" + name + ".tex")


# ------------------------------------------------------------------ model 1
def t_model1():
    r, ws = load("results/model1.json"), load("results/model1_warmstart.json")
    rows = []
    for L in sorted(map(int, r["systems"])):
        s = r["systems"][str(L)]
        vr = s["vqe"]
        Pbest_r = min(vr, key=lambda p: vr[p]["E"])
        if str(L) in ws["systems"]:
            lad = ws["systems"][str(L)]["ladder"]
            Pbest_w = min(lad, key=lambda p: lad[p]["E"])
            wcol = f"{Pbest_w} & {abs(lad[Pbest_w]['rel_err']):.2e}"
        else:
            wcol = "-- & --"
        rows.append(
            f"{L} & {s['dim']} & {s['dim_sz0']} & {s['E0']:.8f} & {s['E0_per_site']:.6f} & "
            f"{Pbest_r} & {abs(vr[Pbest_r]['rel_err']):.2e} & {wcol} \\\\"
        )
    body = (
        "\\begin{tabular}{r r r r r c c c c}\n\\toprule\n"
        "$L$ & $2^L$ & $\\dim_{S^z=0}$ & $E_0$ (ED) & $E_0/L$ & "
        "$P$ & rel.\\ err. & $P$ & rel.\\ err. \\\\\n"
        "& & & & & \\multicolumn{2}{c}{random init} & "
        "\\multicolumn{2}{c}{warm start} \\\\\n\\midrule\n"
        + "\n".join(rows)
        + "\n\\bottomrule\n\\end{tabular}\n"
    )
    w("model1_main", body)

    ext = r["extrapolation"]
    w("model1_extrap",
      f"{ext['E_inf_fit']:.6f},\\; {r['E_inf_exact']:.6f},\\; {abs(ext['deviation_from_bethe']):.1e}")


# ------------------------------------------------------------------ model 2
def t_model2():
    r = load("results/model2.json")
    rows = []
    for L in sorted(map(int, r["systems"])):
        s = r["systems"][str(L)]
        j = s["jastrow_with_phase"]
        rows.append(
            f"{L} & {s['dim_sz0']} & {s['E0_ED']:.10f} & {s['E0_formula']:.10f} & "
            f"{s['formula_abs_err']:.1e} & {j['eigen_residual']:.1e} & "
            f"{j['overlap_with_gs']:.12f} \\\\"
        )
    body = (
        "\\begin{tabular}{r r r r c c c}\n\\toprule\n"
        "$L$ & $\\dim_{S^z=0}$ & $E_0$ (ED) & $-\\frac{\\pi^2}{24}(L+\\frac{5}{L})$ & "
        "$|\\Delta E|$ & $\\|H\\psi-E\\psi\\|$ & $|\\langle \\psi_0|\\psi_{\\rm J}\\rangle|$ \\\\\n"
        "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"
    )
    w("model2_main", body)


# ------------------------------------------------------------------ model 3
def t_model3():
    r = load("results/model3.json")
    c = r["counting"]
    rows = []
    for N in sorted(map(int, c)):
        s = c[str(N)]
        mw = f"{s['mean_weight']:.1f}" if "mean_weight" in s else "--"
        gc = f"{s['n_groups_commuting']}" if "n_groups_commuting" in s else "--"
        gq = f"{s['n_groups_qwc']}" if "n_groups_qwc" in s else "--"
        rows.append(f"{N} & {s['n_qubits']} & {s['hilbert_dim']} & {s['n_terms']} & "
                    f"{mw} & {gc} & {gq} \\\\")
    w("model3_counting",
      "\\begin{tabular}{r r r r c r r}\n\\toprule\n"
      "$N$ & $n=N/2$ & $2^n$ & $\\binom{N}{4}$ & mean weight & "
      "comm.\\ groups & QWC groups \\\\\n\\midrule\n"
      + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

    if "exact_diagonalisation" in r:
        ed = r["exact_diagonalisation"]
        rows = []
        for N in sorted(map(int, ed)):
            s = ed[str(N)]
            rows.append(
                f"{N} & {s['n_qubits']} & {s['n_realisations']} & "
                f"{s['E0_per_majorana']:.6f} & {s['E0_std']/N:.6f} & "
                f"{s['S_half_mean']:.4f} & {100*s['S_over_maximal']:.1f}\\,\\% \\\\"
            )
        w("model3_ed",
          "\\begin{tabular}{r r r r r r r}\n\\toprule\n"
          "$N$ & $n$ & samples & $\\langle E_0\\rangle/N$ & $\\sigma(E_0)/N$ & "
          "$S_{L/2}$ (nats) & of maximal \\\\\n\\midrule\n"
          + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

    if "naive_vqe" in r:
        rows = []
        for N in sorted(map(int, r["naive_vqe"])):
            s = r["naive_vqe"][str(N)]
            sw = s["sweep"]
            best = min(sw, key=lambda p: sw[p]["E"])
            rows.append(f"{N} & {N//2} & {s['E0']:.6f} & {best} & "
                        f"{sw[best]['n_params']} & {sw[best]['E']:.6f} & "
                        f"{100*abs(sw[best]['rel_err']):.1f}\\,\\% \\\\")
        w("model3_naive",
          "\\begin{tabular}{r r r c r r r}\n\\toprule\n"
          "$N$ & $n$ & $E_0$ (ED) & best $P$ & params & $E_{\\rm VQE}$ & rel.\\ err. \\\\\n"
          "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

    if "barren_plateau" in r:
        bp = r["barren_plateau"]
        rows = [f"{N} & {bp[str(N)]['n_qubits']} & {bp[str(N)]['grad_var']:.3e} & "
                f"{bp[str(N)]['grad_abs_mean']:.3e} \\\\" for N in sorted(map(int, bp))]
        w("model3_bp",
          "\\begin{tabular}{r r r r}\n\\toprule\n"
          "$N$ & $n$ & Var$[\\partial_{\\theta_0}E]$ & "
          "$\\mathbb{E}|\\partial_{\\theta_0}E|$ \\\\\n\\midrule\n"
          + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def t_model3b():
    r = load("results/model3b.json")
    rows = []
    for N in sorted(map(int, r)):
        s = r[str(N)]
        rb = s["real_brickwork"]
        cb = s["complex_brickwork"]
        Pr, Pc = max(rb, key=int), max(cb, key=int)
        ad = s["adapt"]["complex_reference"]
        adr = s["adapt"]["real_reference"]
        rows.append(
            f"{N} & {N//2} & {s['gs_imaginary_weight']:.3f} & "
            f"{100*rb[Pr]['rel_err']:.1f} & {100*cb[Pc]['rel_err']:.1f} & "
            f"{adr['n_ops']} & {100*adr['rel_err']:.1f} & "
            f"{ad['n_ops']} & {100*ad['rel_err']:.1f} \\\\"
        )
    w("model3b",
      "\\begin{tabular}{r r c r r c r c r}\n\\toprule\n"
      " & & imag.\\ weight & RY+CZ & RY+RZ+CZ & \\multicolumn{2}{c}{ADAPT, real ref.} & "
      "\\multicolumn{2}{c}{ADAPT, complex ref.} \\\\\n"
      "$N$ & $n$ & of $\\psi_0$ & err.\\ (\\%) & err.\\ (\\%) & ops & err.\\ (\\%) & "
      "ops & err.\\ (\\%) \\\\\n\\midrule\n"
      + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


# ------------------------------------------------------------------ qiskit
def t_qiskit():
    r = load("results/qiskit_crosscheck.json")
    rows = []
    for k, v in r.items():
        if not k.startswith("L"):
            continue
        L = k.split("P")[0][1:]
        P = k.split("P")[1]
        rows.append(f"{L} & {P} & {v['E_inhouse']:.10f} & {v['E_qiskit']:.10f} & "
                    f"{v['abs_diff']:.1e} & {v['overlap']:.12f} & "
                    f"{v['depth_transpiled_cx_basis']} & {v['n_cx']} \\\\")
    w("qiskit_check",
      "\\begin{tabular}{r c r r c c r r}\n\\toprule\n"
      "$L$ & $P$ & $E$ (this work) & $E$ (Qiskit) & $|\\Delta|$ & "
      "$|\\langle\\psi|\\psi\\rangle|$ & depth & CX \\\\\n\\midrule\n"
      + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")




def t_model4():
    r = load("results/model4_extras.json")
    if "shot_noise" in r:
        label = {"XXX_L8": "XXX, $L=8$", "HS_L8": "Haldane--Shastry, $L=8$",
                 "SYK_N8": "SYK, $N=8$", "SYK_N12": "SYK, $N=12$"}
        rows = []
        for k, v in r["shot_noise"].items():
            rows.append(f"{label.get(k,k)} & {v['n_terms']} & {v['l1_norm']:.2f} & "
                        f"{v['n_qwc_groups']} & {v['analytic_worst_case_shots']:.1e} & "
                        f"{v['empirical_total_shots_for_1e-3']:.1e} \\\\")
        w("model4_shots",
          "\\begin{tabular}{l r r r r r}\n\\toprule\n"
          "model & terms & $\\|c\\|_1$ & QWC groups & worst case & measured \\\\\n"
          "& & & & \\multicolumn{2}{c}{shots for $\\epsilon=10^{-3}$} \\\\\n"
          "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    if "gaussian_warm_start" in r:
        rows = []
        for N in sorted(map(int, r["gaussian_warm_start"])):
            v = r["gaussian_warm_start"][str(N)]
            rows.append(
                f"{N} & {N//2} & {v['overlap_q2_with_syk_gs']:.3f} & "
                f"{100*v['cold_start']['rel_err']:.1f} & "
                f"{100*v['adiabatic_path']['rel_err']:.1f} & "
                f"{v['adapt_from_q2']['n_ops']} & "
                f"{100*v['adapt_from_q2']['rel_err']:.2f} \\\\")
        w("model4_warmstart",
          "\\begin{tabular}{r r c r r c r}\n\\toprule\n"
          " & & $|\\langle\\psi_0^{\\rm SYK}|\\psi_0^{q=2}\\rangle|$ & "
          "brickwork & brickwork & \\multicolumn{2}{c}{ADAPT from $\\psi_0^{q=2}$} \\\\\n"
          "$N$ & $n$ & & cold (\\%) & adiabatic (\\%) & ops & err.\\ (\\%) \\\\\n"
          "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


if __name__ == "__main__":
    for fn in (t_model1, t_model2, t_model3, t_model3b, t_model4, t_qiskit):
        try:
            fn()
        except Exception as exc:
            print(f"  skipped {fn.__name__}: {exc}")
