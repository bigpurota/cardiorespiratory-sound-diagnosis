"""
src/split.py — leakage-safe patient-level train/test splits (DATA-03).

Zero patient leakage is the #1 correctness requirement of this project. Every
downstream experiment script (Phase 3+) imports ``assert_no_patient_leakage`` and
logs disjointness at startup.

Heart (D-10): seeded ``GroupShuffleSplit(test_size=0.20, random_state=42)`` computed
WITHIN databases A–E (patient_id = DB-prefixed recording stem). The never-released
private test set is not touched.

Lung (D-03/D-04, Open Question 1): fetch the official ICBHI split, validate it, REPAIR
the 2 patients (156, 218) that the official file places on BOTH sides by forcing all
their recordings to train, assert disjoint, and log the provenance. If the fetch/
validation fails, fall back to a seeded patient-level ``GroupShuffleSplit`` 60/40.

Code derived from 02-RESEARCH.md Code Examples §4 (make_heart_splits), §5
(assert_no_patient_leakage), and §8 (fetch-then-fallback + repair).
"""
import os
import sys

import config  # import FIRST — seeds RNGs, exposes paths (config.py)

sys.path.insert(0, config.PROJECT_ROOT)  # allow `import scripts.fetch_icbhi_split`

DEFAULT_MANIFEST = os.path.join(config.DATA_PROCESSED, "manifest.csv")
HEART_SPLITS_CSV = os.path.join(config.SPLITS_DIR, "heart_splits.csv")
LUNG_SPLITS_CSV = os.path.join(config.SPLITS_DIR, "lung_splits.csv")
LUNG_PROVENANCE = os.path.join(config.SPLITS_DIR, "lung_split_provenance.txt")

# VERIFIED (02-RESEARCH §8 / Pitfall 1): the official ICBHI split places these 2
# patients in BOTH train and test. Force all their recordings to the TRAIN side.
OVERLAP_PATIENTS = {"156", "218"}

# CinC 2016 training databases (D-10): heart split is computed within these only.
HEART_DBS = {"a", "b", "c", "d", "e"}


# ---------------------------------------------------------------------------
# §5 — Importable zero-leakage assertion (reusable by every experiment script)
# ---------------------------------------------------------------------------
def assert_no_patient_leakage(train_ids, test_ids):
    """Raise AssertionError on any patient overlap; log counts on disjoint sets.

    Used at the startup of every Phase-3+ experiment script. On disjoint sets it
    prints the ``[leakage-check OK] ...`` line that surfaces in the Methods section
    (D-04) and returns ``None``.
    """
    train, test = set(map(str, train_ids)), set(map(str, test_ids))
    overlap = train & test
    assert not overlap, (
        f"PATIENT LEAKAGE: {len(overlap)} shared ids e.g. {sorted(overlap)[:5]}"
    )
    print(
        f"[leakage-check OK] train_patients={len(train)} "
        f"test_patients={len(test)} overlap=0"
    )


# ---------------------------------------------------------------------------
# §4 — Heart patient-level split (CinC, WITHIN A–E), deterministic at seed=42
# ---------------------------------------------------------------------------
def make_heart_splits(
    manifest_csv=DEFAULT_MANIFEST,
    out_csv=HEART_SPLITS_CSV,
    test_size=0.20,
    seed=42,
):
    """Build the seeded GroupShuffleSplit heart split WITHIN databases A–E (D-10).

    Writes a de-duplicated patient-level CSV (patient_id, db_source, split) and
    asserts zero patient leakage on the saved split. Deterministic at seed=42.
    """
    import pandas as pd
    from sklearn.model_selection import GroupShuffleSplit

    df = pd.read_csv(manifest_csv)
    df = df[df.modality == "heart"].copy()
    df["db_source"] = df["db_source"].astype(str)
    # D-10: compute the split WITHIN A–E only; never merge/re-split across the
    # never-released private test. Each DB grouped by its patient_id.
    df = df[df["db_source"].isin(HEART_DBS)].copy()
    df["patient_id"] = df["patient_id"].astype(str)

    df["split"] = "train"
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    _, te_idx = next(gss.split(df, groups=df["patient_id"]))
    df.iloc[te_idx, df.columns.get_loc("split")] = "test"

    out = df[["patient_id", "db_source", "split"]].drop_duplicates()
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    out.to_csv(out_csv, index=False)

    assert_no_patient_leakage(
        out[out.split == "train"].patient_id,
        out[out.split == "test"].patient_id,
    )
    return out


# ---------------------------------------------------------------------------
# §8 — Lung patient-level split (official ICBHI + repair, else fallback)
# ---------------------------------------------------------------------------
def make_lung_splits(
    manifest_csv=DEFAULT_MANIFEST,
    out_csv=LUNG_SPLITS_CSV,
    provenance_path=LUNG_PROVENANCE,
    test_size=0.40,
    seed=42,
):
    """Build the lung split: official ICBHI (validated) + 156/218 repair, else fallback.

    Official path: adopt official recording assignments, derive patient_id from the
    recording stem, then REPAIR — force every recording of patients 156 and 218 to the
    train side (Open Question 1 / Pitfall 1). Fallback path: seeded patient-level
    ``GroupShuffleSplit(test_size=0.40, random_state=42)``. Either way: assert disjoint,
    write the CSV (patient_id, split) and a provenance line recording the path taken.
    """
    import pandas as pd

    from scripts.fetch_icbhi_split import fetch_official_split

    manifest = pd.read_csv(manifest_csv)
    lung = manifest[manifest.modality == "lung"].copy()
    lung["patient_id"] = lung["patient_id"].astype(str)

    rows, source_url = fetch_official_split()

    if rows is not None:
        # Official path: stem<TAB>train|test -> patient_id = stem.split('_')[0]
        rec = pd.DataFrame(rows, columns=["stem", "split"])
        rec["patient_id"] = rec["stem"].str.split("_").str[0].astype(str)
        rec["split"] = rec["split"].astype(str)

        out = rec[["patient_id", "split"]].copy()
        # REPAIR: force ALL recordings of overlap patients (156, 218) to train, so each
        # patient lands on exactly ONE side and the disjoint assertion passes.
        repair_mask = out["patient_id"].isin(OVERLAP_PATIENTS)
        out.loc[repair_mask, "split"] = "train"
        out = out.drop_duplicates(subset=["patient_id"])

        provenance = (
            "official-split + 2-patient repair (156,218->train)\n"
            f"source: {source_url}\n"
            "canonical: Harvard Dataverse DOI 10.7910/DVN/HT6PKI\n"
            "2 patients (156, 218) reassigned to train for strict patient-level "
            "integrity; deviates from official split by 2 patients (Open Question 1, "
            "Pitfall 1).\n"
        )
    else:
        # Fallback: seeded patient-level GroupShuffleSplit 60/40 (D-03).
        from sklearn.model_selection import GroupShuffleSplit

        lung["split"] = "train"
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        _, te_idx = next(gss.split(lung, groups=lung["patient_id"]))
        lung.iloc[te_idx, lung.columns.get_loc("split")] = "test"
        out = lung[["patient_id", "split"]].drop_duplicates()

        provenance = (
            "reconstructed seeded GroupShuffleSplit 60/40 "
            f"(test_size={test_size}, random_state={seed})\n"
            "official ICBHI split fetch/validation failed — fell back to seeded "
            "patient-level GroupShuffleSplit (D-03 fallback).\n"
        )

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    out.to_csv(out_csv, index=False)
    with open(provenance_path, "w", encoding="utf-8") as fh:
        fh.write(provenance)

    assert_no_patient_leakage(
        out[out.split == "train"].patient_id,
        out[out.split == "test"].patient_id,
    )
    return out
