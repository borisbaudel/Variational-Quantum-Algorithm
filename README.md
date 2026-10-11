# VQE ground states of three spin-chain Hamiltonians

XXX Heisenberg (PBC) · Haldane–Shastry · Sachdev–Ye–Kitaev

Everything in `report.pdf` is reproduced by the scripts here. No number in the
report is typed by hand: the tables are generated from the JSON output of the
runs by `make_tables.py`, and the figures by `make_figures.py`.

## Layout

```
spinvqe/
  core.py               Pauli algebra, the three Hamiltonians (sparse),
                        S^z-sector projection, exact diagonalisation
  ansatz.py             state-vector engine; the block-structured exchange gate;
                        the Hamiltonian Variational Ansatz with analytic
                        (adjoint) gradients; hardware-efficient brickwork
  vqe.py                optimiser drivers, finite-difference gradients,
                        gradient-variance (barren-plateau) diagnostic
  syk.py                Jordan-Wigner Majorana mapping, symbolic Pauli-string
                        algebra, fast O(2^n) Pauli-string assembly,
                        commuting / qubit-wise-commuting measurement grouping
  qiskit_interface.py   the same objects in Qiskit (SparsePauliOp + circuits)
                        and the cross-validation against the engine above
  analysis.py           generalised parameter-shift rules; an unbiased
                        shot-noise estimator with qubit-wise-commuting groups;
                        the free-fermion (q = 2) Majorana model used as a
                        Gaussian warm start

run_model1.py           Model 1, random initialisation
run_model1_warmstart.py Model 1, layerwise warm start
run_model2.py           Model 2: the sin^2 identity, the closed-form energy,
                        and the exact Gutzwiller-Jastrow ground state
run_model3.py           Model 3: term counting, grouping, exact spectra,
                        entanglement, barren plateaus, naive VQE
run_model3b.py          Model 3: is the ground state complex?  a fair
                        (complex-capable) control, and ADAPT-VQE
run_model3b_quick.py    the same with the depth sweep trimmed to P = 4, so the
                        ADAPT comparison reaches N = 14 in minutes
run_model4_extras.py    shot-noise cost of one energy evaluation, per model;
                        and the q = 2 Gaussian warm start for SYK
verify_derivations.py   all 47 identities of the report's appendix, evaluated
merge_ws_log.py         utility: the warm-start ladder writes its JSON once per
                        system size, so if the (expensive) L = 16 run is stopped
                        mid-ladder this re-reads the completed rungs from the
                        run's own log. No point is invented and none discarded.

make_tables.py          results/*.json -> tables/*.tex
make_figures.py         results/*.json -> figures/*.pdf
tests/test_core.py      18 checks, one per claim made in the report
report.tex, appendix.tex   the write-up (16 pages; the appendix derives every
                        identity and cross-references it to a numerical check)
```

## Reproducing

```bash
pip install numpy scipy matplotlib networkx qiskit
python tests/test_core.py          # 18 tests, ~30 s

python run_model1.py               # ~8 min  (L = 4,6,8,12,16)
python run_model1_warmstart.py     # ~25 min (L = 16 dominates)
python run_model2.py               # ~1 min
python run_model3.py               # ~15 min
python run_model3b.py              # ~10 min

python run_model3b_quick.py        # ~20 min, extends the ADAPT sweep
python run_model4_extras.py        # ~40 min, shot noise + Gaussian warm start
python verify_derivations.py       # ~2 min, 47/47 identities

python merge_ws_log.py             # only if a ladder was cut short
python make_tables.py && python make_figures.py
pdflatex report.tex && pdflatex report.tex
```

Written with numpy 2.4, scipy 1.17, qiskit 2.5, Python 3.11.

## Conventions worth knowing before reading the code

**Pauli, not spin.** The assignment writes the Hamiltonians with Pauli
matrices, and `sigma = 2 S`, so `sigma_i . sigma_j = 4 S_i . S_j`. Every
comparison with a textbook result carries that factor of four. For the XXX
chain the Bethe-ansatz limit is therefore `E/L -> 1 - 4 ln 2 = -1.7725887 J`.

**Endianness.** `core.py` puts site 0 in the most significant tensor factor;
Qiskit is little-endian. All labels handed to Qiskit are reversed, and
`qiskit_interface.crosscheck` verifies that the two give identical expectation
values (agreement ~5e-15, state overlap 1.000000000000).

**Why a hand-written state-vector engine.** `exp(-i theta sigma.sigma)` is the
identity on {|00>,|11>} and `-1 + 2X` on {|01>,|10>}; exploiting that block
structure, plus reverse-mode (adjoint) gradients, is what brings L = 16 with
P = 10 layers down to minutes. Qiskit is used to validate the engine, not to
run the optimisations.

## The short version of the results

**Model 1 — solved.** An HVA that is exactly implementable (disjoint bonds, so
no Trotter error) and protected by SU(2) symmetry, from a product-of-singlets
reference. Machine precision at L = 6 with 3 layers, `1.3e-6` at L = 8.
The useful finding is the *difference between initialisation strategies*:
random starts stall near a few percent at L = 12, 16 regardless of depth,
while growing the circuit one layer at a time from the previous optimum
recovers a factor of 10 at L = 12 and a factor of **105** at L = 16
(1.6e-2 -> 1.5e-4), for no extra parameters.

**Model 2 — solved analytically, which is the honest answer.**
`|z_j - z_k|^2 = 4 sin^2(pi(j-k)/L)` turns the Hamiltonian into the textbook
Haldane–Shastry chain; the closed form `E_0 = -(pi^2 J/24)(L + 5/L)` then
matches exact diagonalisation to ~1e-15 at every L, and the
Gutzwiller–Jastrow state `prod_i z_{n_i} prod_{i<j} (z_{n_i}-z_{n_j})^2` is
verified to be an exact eigenvector with overlap 1 with the numerical ground
state. Zero variational parameters. Dropping the `prod z` momentum factor
still gives an exact eigenvector — of an excited state, with zero overlap with
the ground state; that detail is invisible analytically and obvious
numerically, which is why both were run.

**Model 3 — not solved, with the reasons measured.**
Five obstructions, with different cures:

1. `C(N,4)` Pauli strings (1820 at N = 16, 35960 at N = 32). Greedy colouring
   packs the 1820 into 167 commuting groups but 549 qubit-wise-commuting ones,
   so general commutation is worth a further factor of three.
2. Mean Pauli weight 5.1 out of 8 qubits: the Jordan-Wigner strings make every
   term global, so there is no locality to exploit.
3. The ground state is volume-law entangled (76–93 % of the maximal half-cut
   entropy, growing linearly in n, against logarithmic growth for XXX). This
   is an expressivity statement: no constant-depth ansatz can get there, and
   no optimiser tuning changes that.
4. The ground state is also *intrinsically complex* — residual imaginary
   weight ~0.49 after removing the best global phase. The standard RY+CZ
   "hardware-efficient" ansatz has real amplitudes only, so it saturates at
   the same energy for every depth (identical to 6 digits across P = 2, 4, 8).
   That is a representability failure, not a barren plateau, and separating the
   two is the point of measuring both. Adding one RZ per qubit per layer makes
   the same circuit *exact* at N = 8 and N = 10.
5. Flat gradients: Var[dE/dtheta] falls by a factor ~2.6 between n = 4 and
   n = 8 — consistent with a barren plateau, but too narrow a range to fit a
   slope anyone should believe, so no slope is quoted.

ADAPT-VQE with the Hamiltonian's own quartic terms as the operator pool is
implemented, and from a *product* reference the result is **negative**: with a budget of `6n` operators it
crushes the real-amplitude brickwork but does *not* beat a complex-capable one
(12.1 % vs 8.5 % at N = 12; 27.1 % vs 19.4 % at N = 14). That is reported
rather than tuned away.

What the sweep does show cleanly is where the expressivity wall sits: the
complex brickwork is *exact* at n = 4 and 5, then degrades to 8.5 % at n = 6
and 19.4 % at n = 7 at fixed depth — while the real one sits at 41–54 % at
every size and every depth. Two different failure modes, separated only because
both were measured.

| N | n | imag. weight of psi_0 | RY+CZ | RY+RZ+CZ | ADAPT (real ref.) | ADAPT (complex ref.) |
|---|---|---|---|---|---|---|
| 8 | 4 | 0.494 | 41.7 % | **0.0 %** | 73.7 % (stalled, 1 op) | 4.4 % (24 ops) |
| 10 | 5 | 0.658 | 54.2 % | **0.0 %** | 3.4 % (30 ops) | 4.2 % (30 ops) |
| 12 | 6 | 0.674 | 51.5 % | 8.5 % | 12.1 % (36 ops) | 14.8 % (36 ops) |
| 14 | 7 | 0.627 | 53.2 % | 19.4 % | 27.1 % (42 ops) | 25.3 % (42 ops) |

One instance-specific curiosity, kept because it cost an hour: at N = 8, ADAPT
from a *real* product reference stalls after a single operator with every pool
gradient at machine zero — trapped in the real subspace, and the pool that
generates the trap cannot select its way out. A complex reference unblocks it.
At N = 10, 12, 14 the same real reference does not stall, so it is a property
of that instance, not a general obstruction. Useful as a diagnostic: a run that
dies after one operator with zero gradients is a symmetry, not a barren
plateau, and the fix is the reference state.


## The mathematical appendix, the sampling cost, and the warm start

**A full mathematical appendix, with every identity checked.** 47 derivations —
the Pauli/spin normalisation, the spectrum of the exchange operator from the
swap operator, the closed form of the gate, the SU(2) invariance of the ansatz,
the adjoint gradient, the sin² identity, the Haldane–Shastry reduction and
Jastrow ground state, the Jordan–Wigner algebra in three cases, the hermiticity
sign (−1)^C(q,2), the commutation criterion, the ADAPT selection gradient, the
shot-budget bound — each tagged (`A.1`, `B.5`, …) and each *evaluated* by
`verify_derivations.py`. 47/47 pass; the table of residuals is in the report.

One of those checks catches a trap worth naming: the parameter-shift
rule for the exchange gate does exist. Since σ·σ has eigenvalues +1 and −3, the
single frequency is Ω = 4 and `dE/dθ = 2[E(θ+π/8) − E(θ−π/8)]` is exact for one
bond. What fails is the *naive* π/2-shift version, and for a whole sub-layer of
R disjoint bonds the frequencies are 4, 8, …, 4R so an exact rule needs 2R
evaluations — at L = 8 the two-term rule is wrong by 2.2 in absolute terms.
That count is the actual justification for using adjoint differentiation.

**Sampling cost, measured rather than bounded.** An unbiased grouped estimator
(rotate each qubit into its group's basis, sample bitstrings from |ψ|², keep the
true within-group correlations):

| model | terms | ‖c‖₁ | QWC groups | shots for ε = 10⁻³ |
|---|---|---|---|---|
| XXX, L = 8 | 24 | 24.0 | 3 | 5.6e7 |
| Haldane–Shastry, L = 8 | 84 | 9.7 | 3 | 1.7e6 |
| SYK, N = 8 | 70 | 6.3 | 41 | 3.2e7 |
| SYK, N = 12 | 495 | 23.4 | 196 | 2.7e8 |

The ordering is set by ‖c‖₁, not by the term count — the XXX chain has the
fewest strings and the largest budget. Both spin chains collapse to **three**
groups at any L (all the XX, all the YY, all the ZZ); SYK, being structureless,
needs 196 at N = 12.

**The Gaussian (q = 2) warm start — the one strategy that works.**

| N | n | overlap with SYK gs | brickwork cold | ADAPT from product | **ADAPT from q=2** |
|---|---|---|---|---|---|
| 10 | 5 | 0.43 | 0.01 % | 4.2 % | **0.00 %** |
| 12 | 6 | 0.07 | 8.50 % | 14.8 % | **0.13 %** |
| 14 | 7 | 0.09 | 19.4 % | 25.3 % | 18.7 % |

Note that overlap is *not* the mechanism: the warm start helps most at N = 12,
where the overlap is smallest. What it supplies is a state with the right kind
of entanglement, so the circuit builds corrections rather than everything. The
advantage erodes by N = 14, most likely because the operator budget is fixed at
6n while the entanglement to represent grows in the exponent — stated rather
than hidden.

The practical ordering that came out of all this: **fix the reference state
before touching the ansatz, and fix the ansatz before touching the optimiser.**
Taken the other way round, all one learns is that the optimiser has a hard time.
