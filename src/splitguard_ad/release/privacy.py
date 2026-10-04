"""The release boundary in code: key stretching for participant identifiers.

A single-round digest of an ADNI PTID is invertible by enumeration in under a
second, because the identifier space is roughly ten million. Key stretching is
therefore required rather than merely prudent, and the iteration count is a
parameter of the privacy claim the manuscript makes, so it is pinned here and
asserted by the test suite instead of left to a library default.

The salt is never stored in the repository. It is read from the environment,
has no default, and a release records only its fingerprint.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache

__all__ = ["hash_subject_id", "PBKDF2_ITERATIONS", "PBKDF2_DKLEN", "SALT_ENV_VAR"]

SALT_ENV_VAR = "SPLITGUARD_RELEASE_SALT"


PBKDF2_ITERATIONS = 600_000
PBKDF2_DKLEN = 32


@lru_cache(maxsize=None)
def hash_subject_id(subject_id: str, salt: str) -> str:
    """PBKDF2-HMAC-SHA256 of the subject_id under a secret per-release salt.

    Key stretching is load-bearing, not decorative: ADNI PTIDs occupy a
    ~10^7 space, so a single-round digest would be enumerable in seconds.

    The derivation is deterministic, so it is memoised: releasing five seeds of
    the same cohort otherwise re-derives each of the 220 participants five
    times at 600,000 iterations apiece, which costs minutes and changes
    nothing. The cache lives for one process and never touches disk.
    """
    derived = hashlib.pbkdf2_hmac(
        "sha256",
        subject_id.encode("utf-8"),
        salt.encode("utf-8"),
        PBKDF2_ITERATIONS,
        dklen=PBKDF2_DKLEN,
    )
    return derived.hex()
