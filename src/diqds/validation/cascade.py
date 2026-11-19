"""Cascade information reconciliation (Brassard-Salvail), one-way in effect: Bob corrects his string towards Alice's.
Every parity Alice reveals is counted as leaked information.  Block size of pass 1: k1 = 0.73 / QBER, doubled per pass.
"""
from __future__ import annotations

from collections import deque
from typing import Tuple

import numpy as np


def cascade(alice: np.ndarray, bob: np.ndarray, qber: float, rng: np.random.Generator, passes: int = 4) -> Tuple[np.ndarray, int]:
    n = int(alice.size)
    bob = bob.copy()
    if n == 0:
        return bob, 0
    leak = 0
    k1 = int(max(4, min(n, round(0.73 / max(qber, 1e-4)))))
    perms, sizes = [], []
    a8 = alice.astype(np.uint8)

    def parity(arr, idx):
        return int(np.bitwise_xor.reduce(arr[idx])) if idx.size else 0

    def binary(idx):
        nonlocal leak
        while idx.size > 1:
            h = idx.size // 2
            leak += 1
            if parity(a8, idx[:h]) != parity(bob, idx[:h]):
                idx = idx[:h]
            else:
                idx = idx[h:]
        return int(idx[0])

    for it in range(passes):
        k = max(1, min(k1 * 2 ** it, n))
        perm = np.arange(n) if it == 0 else rng.permutation(n)
        inv = np.empty(n, np.int64)
        inv[perm] = np.arange(n)
        perms.append((perm, inv))
        sizes.append(k)
        starts = np.arange(0, n, k)
        pa = np.bitwise_xor.reduceat(a8[perm], starts)
        pb = np.bitwise_xor.reduceat(bob[perm], starts)
        leak += int(starts.size)
        queue = deque((it, int(j)) for j in np.flatnonzero(pa != pb))
        while queue:
            lvl, j = queue.popleft()
            p_, _ = perms[lvl]
            kk = sizes[lvl]
            idx = p_[j * kk:(j + 1) * kk]
            if parity(a8, idx) == parity(bob, idx):
                continue
            pos = binary(idx)
            bob[pos] ^= 1
            for other in range(len(perms)):
                if other != lvl:
                    queue.append((other, int(perms[other][1][pos] // sizes[other])))
    return bob, leak
