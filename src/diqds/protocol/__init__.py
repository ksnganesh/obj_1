"""The DI-QDS protocol of Fan et al. (, Phys. Rev. A 114, 032617 (2026), Sec. II) in its two stages.

    distribution.py   Distribution stage (Sec. II A).  Alice-Bob and Alice-Charlie each run a DI key-generation
                      protocol on separate devices: 1 parameters, 2 data acquisition, 3 sifting, 4 noisy
                      preprocessing, 5 error correction and error verification, 6 parameter estimation,
                      7 privacy amplification.  Output: K_AB (Alice, Bob) and K_AC (Alice, Charlie).
    messaging.py      Messaging stage (Sec. II B).  The keys are cut into n_s-bit segments (X_a = X_c,
                      Y_a = Y_b xor Y_c, Z_a = Z_c); 1 signing of Alice, 2 verification of Bob, 3 certification of
                      Charlie, over a channel that records every message; forgery and repudiation attempts (Sec. III).
    session.py        One session: the distribution stage, then the messaging stage for each document.

The emulator and the privacy-amplification analysis run these stages (validation/pipeline.py, privacy_amplification/no_pa.py);
scripts/run_protocol_session.py runs one session and prints what every step did.
"""
from .distribution import (DistributionResult, LinkPlan, LinkRun, RawData, Reconciled, data_acquisition,  # noqa: F401
                           distribution_stage, error_correction, error_verification, noisy_preprocessing,
                           parameter_estimation, plan_link, privacy_amplification, run_kgp, sifting)
from .messaging import (Channel, ForgeryAttempt, KeySet, Message, SignatureRecord, alice_key_set,  # noqa: F401
                        alice_sign, bits_to_text, bob_verify, charlie_certify, forgery_attempt, key_sets,
                        messaging_stage, partition, text_to_bits)
from .session import Session, run_session  # noqa: F401
