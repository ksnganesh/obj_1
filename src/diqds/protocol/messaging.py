"""Messaging stage of DI-QDS (Fan et al. Sec. II B and Fig. 2; the OTUH-QDS of Yin et al.).

Keys.  After the distribution stage every party cuts its keys into n_s-bit segments.  Signature j uses the j-th
3 n_s-bit block of each link (138 bits per link for n_s = 46):
    Bob      {X_b, Y_b, Z_b}  from K_AB            Charlie  {X_c, Y_c, Z_c}  from K_AC
    Alice    X_a = X_c,   Y_a = Y_b xor Y_c,   Z_a = Z_c   (from her copies of K_AB and K_AC)    partition, key_sets

Steps (one function each; every message goes through a Channel, which keeps the transcript):
 1. signing of Alice          p_a = f(Z_a), an irreducible polynomial of degree n_s; the LFSR-based Toeplitz
                              hash with feedback p_a and initial state X_a gives Dig = H Doc; Sig = Dig xor Y_a.
                              Alice -> Bob (public channel): {Sig, Doc}                                alice_sign
 2. verification of Bob       Bob -> Charlie (authenticated): {Sig, Doc, Y_b}.  Only after Charlie has recorded
                              it: Charlie -> Bob: {X_c, Y_c, Z_c}.  Bob forms KX_b = X_c, KY_b = Y_b xor Y_c,
                              KZ_b = Z_c, the expected digest Sig xor KY_b and the actual digest
                              H(f(KZ_b), KX_b) Doc.  Equal: he accepts and notifies Charlie (Bob -> Charlie:
                              accept); otherwise he rejects and the protocol aborts                    bob_verify
 3. certification of Charlie  KX_c = X_c, KY_c = Y_b xor Y_c (Y_b as forwarded by Bob), KZ_c = Z_c; the same
                              comparison decides whether she accepts                                   charlie_certify
messaging_stage runs steps 1-3 for one document.

Security properties (Sec. III):
  forgery      a dishonest Bob alters {Doc, Sig} before forwarding them; he wins iff Charlie accepts.  Changing
               Doc by Delta and Sig by t works iff t = h(Delta), which needs X_c and Z_c          forgery_attempt
  repudiation  a dishonest Alice wins iff Bob accepts and Charlie rejects.  Both test the same
               {Sig, Doc, Y_b, X_c, Y_c, Z_c}, so this needs a corrupted key-exchange message: eps_rep = 2 eps,
               which Channel(error_rate) lets one see                                SignatureRecord.repudiated
  robustness   with all parties honest both accept unless the keys differ after the distribution stage.

The hash h_{p,s}(M) = M(x) s(x) mod p(x) (the LFSR-Toeplitz family in polynomial form) and f(Z) (minimal polynomial
by Berlekamp-Massey) are in validation/otuh.py and gf2.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..common import N_S, TABLE_I
from ..validation.gf2 import DegenerateKey
from ..validation.otuh import SegmentKeys, cut, digest
from .distribution import DistributionResult

KeySet = SegmentKeys          # (X, Y, Z), each an n_s-bit integer


# ------------------------------------------------------------------------------------------ classical channel
@dataclass
class Message:
    step: str                     # "1 signing", "2 verification", "3 certification"
    sender: str
    receiver: str
    channel: str                  # "public" or "authenticated"
    content: Dict[str, object]


class Channel:
    """Classical channel between the parties; keeps every message in ``transcript``.
    error_rate: probability that a message arrives with one bit of one of its key or signature fields flipped (the
    eps of Fan et al.'s Eqs. (6)-(7), 1e-11 in Table I; 0 by default).  ``fields`` restricts which fields can be
    corrupted.  The channel draws from its own generator, so it never changes the protocol's random draws."""

    def __init__(self, error_rate: float = 0.0, seed: int = 0, fields: Optional[Sequence[str]] = None, n_s: int = N_S):
        self.error_rate = float(error_rate)
        self.rng = np.random.default_rng(seed)
        self.fields = None if fields is None else set(fields)
        self.n_s = n_s
        self.transcript: List[Message] = []
        self.corrupted: List[Tuple[int, str]] = []          # (message index, field)

    def send(self, step: str, sender: str, receiver: str, channel: str, **content) -> Dict[str, object]:
        delivered = dict(content)
        if self.error_rate > 0 and self.rng.random() < self.error_rate:
            ints = [k for k, v in delivered.items() if isinstance(v, int) and not isinstance(v, bool)
                    and (self.fields is None or k in self.fields)]
            if ints:
                k = ints[int(self.rng.integers(len(ints)))]
                delivered[k] = delivered[k] ^ (1 << int(self.rng.integers(self.n_s)))
                self.corrupted.append((len(self.transcript), k))
        self.transcript.append(Message(step, sender, receiver, channel, delivered))
        return delivered


# ------------------------------------------------------------------------------------------- keys per signature
def partition(key_bits: np.ndarray, j: int, n_s: int = N_S) -> KeySet:
    """The j-th 3 n_s-bit segment of a key, cut into (X, Y, Z)."""
    w = 3 * n_s
    seg = key_bits[j * w:(j + 1) * w]
    if seg.size < w:
        raise IndexError(f"key has {key_bits.size} bits: no segment {j}")
    return cut(seg, n_s)


def alice_key_set(alice_ab: KeySet, alice_ac: KeySet) -> KeySet:
    """Alice's signing keys from her copies of K_AB and K_AC: X_a = X_c, Y_a = Y_b xor Y_c, Z_a = Z_c."""
    return KeySet(X=alice_ac.X, Y=alice_ab.Y ^ alice_ac.Y, Z=alice_ac.Z)


def key_sets(dist: DistributionResult, j: int, n_s: int = N_S) -> Tuple[KeySet, KeySet, KeySet]:
    """(Alice, Bob, Charlie) keys for signature j, each from that party's own copy of the distributed keys."""
    alice = alice_key_set(partition(dist.ab.key_alice, j, n_s), partition(dist.ac.key_alice, j, n_s))
    return alice, partition(dist.ab.key_bob, j, n_s), partition(dist.ac.key_bob, j, n_s)


# ------------------------------------------------------------------------------------------------ the three steps
def alice_sign(doc: np.ndarray, alice: KeySet, n_s: int = N_S, n_m: int = TABLE_I.n_m) -> int:
    """Step 1: Sig = H(f(Z_a), X_a) Doc xor Y_a.  Raises DegenerateKey if Z_a lies in a proper subfield
    (probability ~2^-23 for n_s = 46); the segment is then discarded."""
    if np.asarray(doc).size > n_m:
        raise ValueError(f"document of {np.asarray(doc).size} bits: the forging bound (Eq. 5) assumes at most {n_m}")
    return digest(doc, alice.X, alice.Z, n_s) ^ alice.Y


def _digests(sig: int, doc: np.ndarray, KX: int, KY: int, KZ: int, n_s: int) -> Tuple[int, Optional[int]]:
    """(expected digest Sig xor KY, actual digest H(f(KZ), KX) Doc); actual is None for a degenerate KZ."""
    expected = sig ^ KY                              # decrypt the signature with the one-time pad
    try:
        actual = digest(doc, KX, KZ, n_s)            # hash the document with the reconstructed (p, initial state)
    except DegenerateKey:
        actual = None
    return expected, actual


def _bob_digests(sig, doc, bob: KeySet, charlie_revealed: KeySet, n_s: int):
    return _digests(sig, doc, charlie_revealed.X, bob.Y ^ charlie_revealed.Y, charlie_revealed.Z, n_s)


def _charlie_digests(sig, doc, Y_b_forwarded: int, charlie: KeySet, n_s: int):
    return _digests(sig, doc, charlie.X, Y_b_forwarded ^ charlie.Y, charlie.Z, n_s)


def bob_verify(sig: int, doc: np.ndarray, bob: KeySet, charlie_revealed: KeySet, n_s: int = N_S) -> bool:
    """Step 2: KX_b = X_c, KY_b = Y_b xor Y_c, KZ_b = Z_c; accept iff the expected and actual digests agree."""
    expected, actual = _bob_digests(sig, doc, bob, charlie_revealed, n_s)
    return actual == expected


def charlie_certify(sig: int, doc: np.ndarray, Y_b_forwarded: int, charlie: KeySet, n_s: int = N_S) -> bool:
    """Step 3: KX_c = X_c, KY_c = Y_b xor Y_c, KZ_c = Z_c; accept iff the expected and actual digests agree."""
    expected, actual = _charlie_digests(sig, doc, Y_b_forwarded, charlie, n_s)
    return actual == expected


@dataclass
class SignatureRecord:
    index: int
    doc: np.ndarray
    sig: Optional[int]                         # None: Alice could not sign (degenerate Z_a)
    bob_accepts: bool
    charlie_accepts: bool
    transcript: List[Message] = field(default_factory=list)
    note: str = ""
    bob_digests: Optional[Tuple[int, Optional[int]]] = None        # (expected, actual) in step 2
    charlie_digests: Optional[Tuple[int, Optional[int]]] = None    # (expected, actual) in step 3

    @property
    def accepted(self) -> bool:
        """Robustness event: both recipients accept."""
        return self.bob_accepts and self.charlie_accepts

    @property
    def repudiated(self) -> bool:
        """Repudiation event: Bob accepts but Charlie rejects."""
        return self.bob_accepts and not self.charlie_accepts


def messaging_stage(doc: np.ndarray, alice: Optional[KeySet], bob: KeySet, charlie: KeySet,
                    channel: Optional[Channel] = None, n_s: int = N_S, index: int = 0,
                    sig: Optional[int] = None) -> SignatureRecord:
    """Steps 1-3 for one document.  ``sig``: the signature a dishonest Alice sends instead of signing (repudiation
    attempt; ``alice`` may then be None)."""
    ch = channel if channel is not None else Channel(n_s=n_s)
    start = len(ch.transcript)
    # 1. signing of Alice
    if sig is None:
        try:
            sig = alice_sign(doc, alice, n_s)
        except DegenerateKey:
            return SignatureRecord(index, doc, None, False, False, [], "degenerate Z_a: segment discarded")
    to_bob = ch.send("1 signing", "Alice", "Bob", "public", Sig=sig, Doc=doc)
    # 2. verification of Bob: forward first, receive Charlie's keys only after she has recorded the forward
    to_charlie = ch.send("2 verification", "Bob", "Charlie", "authenticated",
                         Sig=to_bob["Sig"], Doc=to_bob["Doc"], Y_b=bob.Y)
    revealed = ch.send("2 verification", "Charlie", "Bob", "authenticated", X_c=charlie.X, Y_c=charlie.Y, Z_c=charlie.Z)
    bd = _bob_digests(to_bob["Sig"], to_bob["Doc"], bob, KeySet(revealed["X_c"], revealed["Y_c"], revealed["Z_c"]), n_s)
    ok_b = bd[1] == bd[0]
    ch.send("2 verification", "Bob", "Charlie", "authenticated", decision="accept" if ok_b else "reject")
    if not ok_b:
        return SignatureRecord(index, doc, sig, False, False, ch.transcript[start:], "Bob rejected: protocol aborted",
                               bob_digests=bd)
    # 3. certification of Charlie, with the {Sig, Doc, Y_b} she recorded
    cd = _charlie_digests(to_charlie["Sig"], to_charlie["Doc"], to_charlie["Y_b"], charlie, n_s)
    return SignatureRecord(index, doc, sig, True, cd[1] == cd[0], ch.transcript[start:], bob_digests=bd,
                           charlie_digests=cd)


# ------------------------------------------------------------------------------------------- forgery (Sec. III A)
@dataclass
class ForgeryAttempt:
    doc: np.ndarray            # the document Bob forwards to Charlie
    sig: int                   # the signature Bob forwards
    knew_charlie: bool         # Bob knew (X_c, Z_c)
    accepted: bool             # Charlie's decision: True = successful forgery


def _pad(a: np.ndarray, n: int) -> np.ndarray:
    out = np.zeros(n, np.uint8)
    out[:a.size] = a
    return out


def forgery_attempt(record: SignatureRecord, bob: KeySet, charlie: KeySet, rng: np.random.Generator,
                    knows: Optional[KeySet] = None, altered_doc: Optional[np.ndarray] = None,
                    n_s: int = N_S) -> ForgeryAttempt:
    """A dishonest Bob intercepts Alice's {Doc, Sig}, changes the document by Delta (one random bit, or into
    ``altered_doc``) and the signature by t, and forwards {Doc xor Delta, Sig xor t, Y_b} to Charlie.  The hash is
    linear in the document, so Charlie accepts iff t = h_{f(Z_c), X_c}(Delta).  With ``knows`` (Charlie's X_c and
    Z_c, e.g. learned in the window attack without privacy amplification) Bob computes t; otherwise he guesses it
    and succeeds with probability 2^-n_s."""
    doc = np.asarray(record.doc, np.uint8)
    if altered_doc is None:
        delta = np.zeros(doc.size, np.uint8)
        delta[int(rng.integers(0, doc.size))] = 1
        doc2 = doc ^ delta
    else:
        m = max(doc.size, np.asarray(altered_doc).size)
        doc2 = _pad(np.asarray(altered_doc, np.uint8), m)
        delta = _pad(doc, m) ^ doc2
    mask = (1 << n_s) - 1
    if knows is not None:
        try:
            t = digest(delta, knows.X, knows.Z, n_s)
        except DegenerateKey:
            t = int(rng.integers(0, 2 ** 62)) & mask
    else:
        t = int(rng.integers(0, 2 ** 62)) & mask
    sig2 = record.sig ^ t
    return ForgeryAttempt(doc2, sig2, knows is not None, charlie_certify(sig2, doc2, bob.Y, charlie, n_s))


# --------------------------------------------------------------------------------------------- documents as text
def text_to_bits(text: str) -> np.ndarray:
    """UTF-8 bytes, most significant bit first."""
    return np.unpackbits(np.frombuffer(text.encode("utf-8"), dtype=np.uint8))


def bits_to_text(bits: np.ndarray) -> str:
    b = np.asarray(bits, np.uint8)
    return np.packbits(b[:b.size - b.size % 8]).tobytes().decode("utf-8", errors="replace")
