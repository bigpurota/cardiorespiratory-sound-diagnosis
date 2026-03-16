"""
scripts/fetch_icbhi_split.py — fetch + VALIDATE the official ICBHI 2017 split (D-03).

Read-only HTTPS fetch of the official ``official_split.txt`` from a public mirror, with
strict input validation before the file is trusted (threat T-05). Returns
``(rows, source_url)`` on a reachable + valid mirror, or ``(None, None)`` so the caller
(src.split.make_lung_splits) falls back to a seeded GroupShuffleSplit. Mirrors the
fetch-with-fallback / fail-clear style of scripts/download_lung.py.

VERIFIED (02-RESEARCH §8): the raymin0223 mirror returns 920 rows of
``stem<TAB>{train|test}``; 2 patients (156, 218) overlap (repaired downstream).
Harvard Dataverse DOI 10.7910/DVN/HT6PKI is cited as the canonical source (D-03).
"""
import pathlib
import sys
import urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import config  # import FIRST (seeds, paths)

# Canonical source: Harvard Dataverse DOI 10.7910/DVN/HT6PKI (cited per CONTEXT D-03).
# Practical fetch: the raymin0223 mirror (VERIFIED reachable, 920 rows, 62/38).
MIRRORS = [
    "https://raw.githubusercontent.com/raymin0223/patch-mix_contrastive_learning/"
    "main/data/icbhi_dataset/official_split.txt",
]

EXPECTED_ROWS = 920
VALID_FLAGS = {"train", "test"}


def _on_disk_wav_stems():
    """Return the set of WAV file stems present under ICBHI2017_DIR (T-05 membership)."""
    icbhi_dir = pathlib.Path(config.ICBHI2017_DIR)
    return {p.stem for p in icbhi_dir.rglob("*.wav")}


def _validate(rows):
    """Reject a fetched split unless 920 rows, flags valid, stems ⊆ on-disk WAVs (T-05).

    Returns True only if every check passes. Any failure -> caller falls back.
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
            "fetched split (cannot confirm membership; T-05).",
            file=sys.stderr,
        )
        return False

    fetched_stems = {r[0] for r in rows}
    extraneous = fetched_stems - disk_stems
    if extraneous:
        print(
            f"[WARN] fetched split references {len(extraneous)} stems not on disk "
            f"(e.g. {sorted(extraneous)[:3]}) — rejecting (T-05).",
            file=sys.stderr,
        )
        return False

    print(
        f"[INFO] official split validated: {len(rows)} rows, flags ⊆ {VALID_FLAGS}, "
        "all stems on disk."
    )
    return True


def fetch_official_split(timeout=15):
    """Fetch + validate the official ICBHI split; return (rows, url) or (None, None).

    rows is ``[[stem, "train"|"test"], ...]``. Returns ``(None, None)`` when every
    mirror is unreachable OR the fetched file fails validation, so the caller falls
    back to a seeded patient-level GroupShuffleSplit (D-03).
    """
    for url in MIRRORS:
        try:
            txt = urllib.request.urlopen(url, timeout=timeout).read().decode()
        except Exception as exc:  # network/HTTP failure -> try next mirror, else fallback
            print(f"[WARN] {url} unreachable: {exc}", file=sys.stderr)
            continue

        rows = [ln.split() for ln in txt.splitlines() if ln.strip()]
        if _validate(rows):
            return rows, url
        # validation failed: do not trust this mirror; try the next one.

    return None, None


if __name__ == "__main__":
    fetched, src = fetch_official_split()
    if fetched is None:
        print("[INFO] no valid official split fetched — caller will fall back.")
    else:
        print(f"[INFO] fetched {len(fetched)} rows from {src}")
