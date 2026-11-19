# Methods: where each part of the article lives in the code

| Article | Module | What it implements |
|---|---|---|
| Sec. II A, Fig. 1 | `src/diqds/protocol/distribution.py` | the seven steps of each key-generation link (parameters, data acquisition, sifting, noisy preprocessing, error correction and verification, parameter estimation, privacy amplification) |
| Sec. II B, Fig. 2 | `src/diqds/protocol/messaging.py`, `validation/otuh.py`, `validation/gf2.py` | OTUH signing, verification, certification; f(Z) by Berlekamp–Massey; forgery and repudiation attempts |
| Sec. II A, Eq. (8) | `src/diqds/finite_size/protocols.py` | two settings (sifted), one setting with a common flag, one setting with independent flags (Lu et al.), pre-shared seed |
| Sec. III A, Table I | `src/diqds/finite_size/eat.py` | the EAT key length of Fan et al. (Eq. 9 = Tan et al. Thm 1), optimiser over all free parameters, exact thresholds |
| Sec. III B, Eqs. (5)–(8), Lemmas 1–2, App. B–C | `src/diqds/finite_size/entropy.py`, `renyi.py` | Hahn et al.'s Rényi bounds with noisy preprocessing, the sifted rate of Lemma 1, the exact inner problem with its dual bound, the certified branch-and-bound over w, the key length (Hahn et al. Eq. D6) |
| Sec. III B–IV, Tables II–IV | `src/diqds/finite_size/optimize.py` | multi-start optimiser (monotone in q), thresholds by Brent's method on integer n, V_min, jitter of the threshold |
| Sec. III B, Appendix B (thresholds) | `src/diqds/finite_size/descent.py` | warm-started descent below a threshold whose key length overshoots the target (used by both threshold searches) |
| Sec. V, Tables V–VI, Figs. 5–6 | `src/diqds/randomness/` | ledger in four currencies (incl. authentication with key recycling), platform models (QuTiP, Qiskit Aer, SeQUeNCe), Pareto fronts |
| Sec. V B, App. E | `src/diqds/finite_size/photonic.py` | photon-number-resolved model of the quantum-dot photonic link (operating points S = 2.631 / 2.450) |
| Sec. VI, Props. 1–2, Theorem 2, Tables VII–X | `src/diqds/privacy_amplification/` | window device and its certified acceptance bound, the i.i.d. CHSH attack, the certification limit (h_min, S_half, q_half), seed options |
| Sec. VII, Tables XI–XII | `src/diqds/validation/` | Aer devices, adversarial devices, Cascade, acceptance statistics, complete runs |

Scripts (one per section) write the cached results in `results/`; `scripts/make_tables.py` turns them into
`results/tables.md` and `results/article_numbers.json`, the source of every number quoted in the article.
