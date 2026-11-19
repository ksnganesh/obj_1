# diqds — certification limits and resource costs of device-independent quantum digital signatures

Code, cached results and tests for the article *Certification limits and resource costs of device-independent quantum
digital signatures* (simulation only: NumPy/SciPy for the bounds, QuTiP, Qiskit Aer and SeQUeNCe for the device and
link models). It analyses the DI-QDS scheme of Fan et al., Phys. Rev. A **114**, 032617 (2026), and answers three
questions:

1. **Can DI-QDS skip privacy amplification?** No, for the variants considered. Two device strategies break raw key
   segments (sequential, and randomly sampled near the first-signature block), and a certification limit (Theorem 2)
   shows that below a CHSH score of 2(2√2 − 1)^1/2 ≈ 2.704, without noisy preprocessing, no argument that goes
   through the min-entropy of a raw segment can certify unhashed signature keys, for any signature length.
2. **How much trusted randomness does a signature cost?** At ten times the first-signature block, about
   130–151 private bits per signature with one key setting and a common test flag on ion and atom platforms,
   775–1,003 with independent test flags and 2,055–2,883 with two sifted settings; photonic links that
   draw inputs per attempt need 2.84e8–2.02e9 without pre-shared secrets. The first signature pays for its
   whole block.
3. **How much data does the first signature need?** With the Rényi entropy-accumulation bounds of Arqand et al. and
   Hahn et al. (whose improvement this is), a run with the reported 11 km single-atom statistics needs 4.97e5
   heralded events (one setting, common flag), against 2.59e6 with the EAT bound used by Fan et al.; the
   recorded block would support 514 signatures with the error-correction budget of our Cascade.
   These are projections from reported statistics, not analyses of raw data.

## Install and test

```bash
uv sync --locked          # Python 3.13 and the locked dependencies
uv run pytest -q          # 68 tests, under a minute
```

## Reproduce the article

Each script recomputes one part of the cached results in `results/`; `make_tables.py` then writes every table of the
article (`results/tables.md`) and every number quoted in its text (`results/article_numbers.json`).

| Article | Command | Output | Time (one core) |
|---|---|---|---|
| Tables I–II, Fig. 3; Table III, Fig. 4; Table XIV | `uv run scripts/run_finite_size.py --section renyi` | `finite_size_renyi.json` | ~45 min |
| Table I (reproduction of Fan et al.), EAT columns | `uv run scripts/run_finite_size.py --section eat` | `finite_size_eat.json` | ~60 min |
| Table IV, Sec. II C, Sec. VIII (82-bit, n_m) | `uv run scripts/run_finite_size.py --section sensitivity` | `finite_size_sensitivity.json` | ~40 min |
| Table XIV (Lu et al.'s protocol at 11 km) | `uv run scripts/run_finite_size.py --section lu` | `finite_size_renyi.json` | ~5 min |
| Sec. III B (reproducibility of thresholds) | `uv run scripts/run_finite_size.py --section jitter` | `finite_size_renyi.json` | ~5 min |
| Refine stored thresholds (Appendix B) | `uv run scripts/run_finite_size.py --section polish` | all three `finite_size_*.json` | ~30 min |
| Figs. 3–4 | `uv run scripts/run_finite_size.py --section figures` | `fig_*.png`, `fig_*.pdf` | seconds |
| Sec. V, Tables V–VI, E1–E2, Figs. 5–6 | `uv run scripts/run_randomness.py` | `randomness_*.json` | ~40 min |
| Sec. VI, Tables VII–X, Figs. 7–9 | `uv run scripts/run_privacy_amplification.py` | `privacy_amplification.json` | ~15 min |
| Sec. VII, Tables XI–XII, Fig. 10 | `uv run scripts/run_validation.py` | `validation.json` | ~25 min |
| Sec. VIII, Table XIII | `uv run scripts/run_comparison.py` | `comparison.json` | seconds |
| One complete session, stage by stage | `uv run scripts/run_protocol_session.py` | `protocol_session.json` | ~1 min |
| Figs. 1–2 | `uv run scripts/make_protocol_figures.py` | `fig_protocol_*.png|pdf` | seconds |
| All tables and quoted numbers | `uv run scripts/make_tables.py` | `tables.md`, `article_numbers.json` | seconds |

Every threshold in `finite_size_*.json` is stored with the certified key length at that block size (`ell_at_n_min`,
138–143 bits for 138-bit targets, 138–160 bits on the 400 m link). The `renyi`, `eat` and `sensitivity` sections already apply the
warm-started descent of `finite_size/descent.py`; `--section polish` applies it to results stored by an earlier run, and
each refined entry keeps its earlier value in `polished_from`.

## Layout

```
src/diqds/
  common.py                 QDS security budget (Table I of Fan et al.), honest-device statistics
  finite_size/              EAT baseline, Rényi bound (exact inner problem, certified outer minimisation),
                            protocols, optimiser and exact thresholds, operating points, photonic model
  randomness/               trusted-randomness ledger, QuTiP device models, SeQUeNCe links, platforms, Pareto fronts
  privacy_amplification/    hash families, window device, i.i.d. attack, Theorem 2, seed options
  validation/               Aer and adversarial devices, Cascade, GF(2) and OTUH hashing, Monte Carlo
  protocol/                 the two stages of the protocol: distribution.py and messaging.py, session.py
scripts/                    one script per section, plus make_tables.py
tests/                      unit tests
results/                    cached results, figures (PNG and PDF), tables.md, article_numbers.json
docs/methods.md             map from the article to the code
```

## The protocol in code

`protocol/distribution.py` implements the distribution stage of Fan et al. (Sec. II A) with one function per step
(`plan_link`, `data_acquisition`, `sifting`, `noisy_preprocessing`, `error_correction`, `error_verification`,
`parameter_estimation`, `privacy_amplification`; `distribution_stage` runs both links). `protocol/messaging.py`
implements the messaging stage (Sec. II B): `alice_sign`, `bob_verify`, `charlie_certify` over a recorded `Channel`,
with `forgery_attempt` and repudiation checks. `scripts/run_protocol_session.py` prints one complete session.
