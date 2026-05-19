"""
Fetch and validate the official ICBHI 2017 train/test split.

Performs a read-only HTTPS fetch of official_split.txt from a public mirror and
validates it before trusting the contents. Returns ``(rows, source_url)`` on a
reachable, valid mirror, or ``(None, None)`` so the caller (src.split.make_lung_splits)
falls back to a seeded GroupShuffleSplit.

The raymin0223 mirror returns 920 rows of ``stem<TAB>{train|test}``; 2 patients
(156, 218) overlap and are repaired downstream. The canonical source is Harvard
Dataverse DOI 10.7910/DVN/HT6PKI.
"""
import pathlib
import sys
import urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src import config  # imported first to seed RNGs and expose paths

# Canonical source is Harvard Dataverse DOI 10.7910/DVN/HT6PKI; in practice we
# fetch the raymin0223 mirror (920 rows, roughly a 62/38 train/test split).
MIRRORS = [
    "https://raw.githubusercontent.com/raymin0223/patch-mix_contrastive_learning/"
    "main/data/icbhi_dataset/official_split.txt",
]

EXPECTED_ROWS = 920
VALID_FLAGS = {"train", "test"}


def _on_disk_wav_stems():
    """Return the set of WAV file stems present under ICBHI2017_DIR."""
    icbhi_dir = pathlib.Path(config.ICBHI2017_DIR)
    return {p.stem for p in icbhi_dir.rglob("*.wav")}


def _validate(rows):
    """Accept a fetched split only if it has 920 rows, valid flags, and stems that
    are all present on disk. Any failure returns False and the caller falls back.
    """
    if rows is None or len(rows) != EXPECTED_ROWS:
        print(
            f"[WARN] fetched split has {0 if rows is None else len(rows)} rows "
            f"(expected {EXPECTED_ROWS}) — rejecting.",
            file=sys.stderr,
        )
        return False

    for r in rows:
        if len(r) != 2 or r[1] not in VALID_FLAGS:
            print(f"[WARN] malformed split row {r!r} — rejecting.", file=sys.stderr)
            return False

    disk_stems = _on_disk_wav_stems()
    if not disk_stems:
        print(
            "[WARN] no on-disk ICBHI WAV stems found to validate against — rejecting "
            "fetched split (cannot confirm membership).",
            file=sys.stderr,
        )
        return False

    fetched_stems = {r[0] for r in rows}
    extraneous = fetched_stems - disk_stems
    if extraneous:
        print(
            f"[WARN] fetched split references {len(extraneous)} stems not on disk "
            f"(e.g. {sorted(extraneous)[:3]}) — rejecting.",
            file=sys.stderr,
        )
        return False

    print(
        f"[INFO] official split validated: {len(rows)} rows, flags ⊆ {VALID_FLAGS}, "
        "all stems on disk."
    )
    return True


def fetch_official_split(timeout=15):
    """Fetch and validate the official ICBHI split; return (rows, url) or (None, None).

    ``rows`` is ``[[stem, "train"|"test"], ...]``. Returns ``(None, None)`` when every
    mirror is unreachable or the fetched file fails validation, so the caller falls
    back to a seeded patient-level GroupShuffleSplit.
    """
    for url in MIRRORS:
        try:
            txt = urllib.request.urlopen(url, timeout=timeout).read().decode()
        except Exception as exc:  # try the next mirror on any network/HTTP failure
            print(f"[WARN] {url} unreachable: {exc}", file=sys.stderr)
            continue

        rows = [ln.split() for ln in txt.splitlines() if ln.strip()]
        if _validate(rows):
            return rows, url
        # validation failed: don't trust this mirror, try the next one

    return None, None


if __name__ == "__main__":
    fetched, src = fetch_official_split()
    if fetched is None:
        print("[INFO] no valid official split fetched — caller will fall back.")
    else:
        print(f"[INFO] fetched {len(fetched)} rows from {src}")
