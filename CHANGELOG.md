# Changelog

## 0.5.0 (2026-11-19) — revision after review

Numerics (the review's Major comments 5 and 6)
- Thresholds are now exact: Brent's method in log n on integer block sizes (relative tolerance 1e-5) replaces the
  bisection with tolerance 2e-3 that returned the upper end of the bracket (up to 0.46 % high).  Where the key length
  at the root still exceeds the target by more than 5 bits (an optimisation just below the root stopped at a poorer
  local optimum), a warm-started descent (`finite_size/descent.py`) walks down to the smallest n that reaches it;
  `run_finite_size.py --section polish` applies the same descent to stored results.  Every threshold is stored with
  the key length at that n (`ell_at_n_min`), a few bits above 138 instead of up to 1,669.
- The infimum over w in h_alpha is certified: a branch-and-bound lower bound (tangent of the convex relative entropy,
  monotone F, weak duality for the inner problem) over the whole quantum range [W_MIN, W_MAX]; reported key lengths use
  it (within 0.01 bits of the grid value).  The inner problem is now solved exactly by its breakpoints.
- Optimiser: coarse-grid seeding, multi-start Nelder-Mead with restart, warm starts across block sizes; the optimised
  preprocessing result is never below the q = 0 (p = 0) optimum, for the Renyi and the EAT bounds alike.
- `optimize.jitter` measures the sawtooth that integer acceptance thresholds put on ell(n) (reproducibility ~1e-4).

New analyses
- `protocols.INDEPENDENT`: the independent-test-flag protocol that Lu et al. ran at 11 km (gamma_A = 0.26,
  gamma_B = 0.13); like-for-like evaluation at 1e-5, 5e-11 and 1e-15 (`finite_size_renyi.json`, section `lu`).
- Lemma 1 generalised to any public sifting probability s (`entropy.sifted_rate`).
- Sensitivity: EC budget of the emulator's Cascade, memory dephasing over the extra L/c wait with the T2 inferred
  from Lu et al. (27.5 ms), document length n_m (n_s = 46, 56, 66 for 1e3, 1e6, 1e9 bits), 82-bit signatures of
  Bian et al., misaligned key observables (`finite_size_sensitivity.json`).
- Theorem 2 (privacy_amplification/no_pa.py): min-entropy per raw bit under the accepted i.i.d. attack, S_half =
  2 sqrt(2 sqrt2 - 1), q_half, and the smallest forging bound any min-entropy argument can give for raw hash keys;
  a certified lower bound on the window device's acceptance probability (coupling).
- Ledger: the first signature pays for its whole block (`block_totals`); authentication keys (Wegman-Carter with key
  recycling) per block and per signature; the independent-flag protocol on all platforms.
- `finite_size/photonic.py`: the photon-number model of the photonic link, which reproduces the photonic operating
  points used in the ledger (S = 2.631 / 2.450, Q = 3.48 % / 6.65 %).

Repository
- New layout: `src/diqds/` (formerly `diqds_o1/`; finite_size, randomness, privacy_amplification, validation, protocol), `scripts/`,
  `tests/`, `results/`, `docs/`; result files and figures renamed by topic; figures as PNG and vector PDF, without
  in-figure titles.  `scripts/make_tables.py` writes `results/tables.md` and `results/article_numbers.json`, the
  source of every number quoted in the article.
- Internal labels removed; URLs point to github.com/ksnganesh/obj_1; `.zenodo.json` for an archived release; the CI
  workflow is part of the repository.
- `make_tables.py`: signature counts are rounded down (only complete signatures count); scientific notation keeps
  three significant figures; the number of collected tests is recorded with the article numbers.
- Figures: no per-bar number labels on the ledger figure; legends that covered curves moved; the seed-economy figure
  starts at the first-signature block of Table III, which pays for the whole seed.

## 0.4.0 — 2026-11-18

* `scripts/make_protocol_figures.py`: schematics of the protocol, drawn from scratch — `fig_protocol_distribution.png`
  (three parties, two independent DI links, where trusted randomness enters; the seven steps of the key-generation
  protocol) and `fig_protocol_messaging.png` (signing, verification, certification in time order, with the forgery and
  repudiation conditions). They are Figs. 1 and 2 of the article.
* `scripts/run_comparison.py` → `results/comparison.json`, Tables C.1–C.2: our results next to five published analyses,
  with the published values and their sources in the script header — Fan et al., Bian et al. (composable
  DI-QDS, arXiv:2610.39360), Nadlinger et al. (ions), Hahn et al. (Rényi re-analysis), Lu et al. (atoms, 11 km). New
  numbers computed there: rounds to the first signature with Bian et al.'s 82-bit signatures (246 key bits per link;
  the threshold moves by at most 0.4 %) and Fan et al.'s EAT bound on the ion record (78,493 bits, 568 signatures,
  against 3,023 with the Rényi bound).

## 0.3.0 — 2026-11-18

The two stages of the protocol are now explicit in the code.

* New package `diqds_o1/protocol/`:
  * `distribution.py` — **distribution stage** (Fan et al. Sec. II A), one function per step of the key-generation
    protocol: `plan_link` (1 parameters), `data_acquisition` (2), `sifting` (3), `noisy_preprocessing` (4),
    `error_correction` and `error_verification` (5), `parameter_estimation` (6), `privacy_amplification` (7);
    `run_kgp` runs one link, `distribution_stage` the A–B and A–C links. Each run records what every step did.
  * `messaging.py` — **messaging stage** (Sec. II B): `partition` / `key_sets` (X_a = X_c, Y_a = Y_b ⊕ Y_c,
    Z_a = Z_c), `alice_sign` (1), `bob_verify` (2), `charlie_certify` (3), `messaging_stage` over a `Channel` that
    records every message in the order of Fan et al.'s Fig. 2; `forgery_attempt` and repudiation (Sec. III).
  * `session.py` — `run_session`: both stages for a list of documents.
* `part7_validation/pipeline.py` and `part6_privacy_amp/no_pa.py` now run the protocol through these modules;
  `otuh.py` keeps only the hash. The forgery uses the linearity of the hash (it needs X_c and Z_c, not Y_c), as in
  Sec. III A of the Fan et al.. Fourteen fixed-seed scenarios give bit-identical results before and after, so every
  cached result and table is unchanged.
* `scripts/run_protocol_session.py`: one complete session on Aer devices calibrated to the 11 km link, printed step by
  step (Table 7.5, `results/protocol_session.json`).
* 12 new tests (`tests/test_protocol.py`), and a hash-linearity test in place of the old OTUH test; 54 in total.

## 0.2.0 — 2026-11-17

Checked against the sources and extended for journal submission.

**Corrections**
* 400 m atom link: the "better key basis" result (3.35·10⁵ rounds) is withdrawn. The experiment measured its QBER in
  two key bases that are not Alice's CHSH settings (7.81 % and 7.77 %), so the second-basis error cannot be inferred
  from S. `ATOM_400M` now uses Q_x = Q; the best-basis row is replaced by an isotropic (Q_x = Q) sensitivity row.
* The EAT-vs-Rényi gain is now reported like for like: 4.5–6.0× fewer rounds with no preprocessing on either side and
  4.8–6.0× with optimised preprocessing on both (version 0.1.0 quoted 5.3–8.0× from EAT p = 0 against Rényi q opt).
* The extra memory wait of the one-setting protocol is L/c (node separation over fibre light speed), as the code
  always computed; the notes said L/2c.

**Additions**
* `Honest.Qx` makes the second-basis error rate an explicit model choice (`Q_x_state` default, `isotropic()`).
* Part 4: `--section fair` (Rényi q = 0, EAT with optimised p, cross-checks against Nadlinger et al. and Hahn et al.),
  `--section sensitivity` (ε_snd 10⁻⁵–10⁻¹⁵, EC efficiency 1.0–1.3, completeness conventions, Fan et al.'s
  two-setting protocol under the EAT at 11 km, bits per event against Lu et al.).
* Part 6: Proposition 2 — the i.i.d. CHSH-optimal attack against randomly sampled no-PA segments, with a QuTiP check
  of the guessing probability and an acceptance Monte Carlo (`no_pa.iid_attack_on_sampling`, `fig_iid_attack.png`,
  Table 6.4); the former Proposition 2 is now Proposition 3.
* Part 7: measured Cascade efficiency at the error rates of the reported links (`--only-ec`, Table 7.4): f = 1.14–1.19
  on the key rounds of the ion and 11 km links and 1.30 on the test rounds, inside the emulator's budget (1.20 / 1.33);
  1.23 at the 7.8 % QBER of the 400 m link.
* CI workflow (`.github/workflows/tests.yml`), `CITATION.cff`, `.DS_Store` ignored, 42 tests.

## 0.1.0

First complete version of Parts 4–7 (the version on GitHub at commit 4029ed2).
