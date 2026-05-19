"""
Leakage-safe patient-level train/test splits.

Avoiding patient leakage is the main correctness requirement here: every downstream
experiment imports ``assert_no_patient_leakage`` and logs disjointness at startup.

Heart: seeded ``GroupShuffleSplit(test_size=0.20, random_state=42)`` computed within
databases A-E (patient_id = DB-prefixed recording stem); the private test set is untouched.

Lung: fetch the official ICBHI split, validate it, then force the two patients (156, 218)
that the official file places on both sides onto the train side, assert disjointness, and
log provenance. If the fetch or validation fails, fall back to a seeded patient-level
``GroupShuffleSplit`` 60/40.
"""
import os
import sys

from src import config  # import first — seeds RNGs and exposes paths

sys.path.insert(0, config.PROJECT_ROOT)  # allow `import scripts.fetch_icbhi_split`

DEFAULT_MANIFEST = os.path.join(config.DATA_PROCESSED, "manifest.csv")
HEART_SPLITS_CSV = os.path.join(config.SPLITS_DIR, "heart_splits.csv")
LUNG_SPLITS_CSV = os.path.join(config.SPLITS_DIR, "lung_splits.csv")
LUNG_PROVENANCE = os.path.join(config.SPLITS_DIR, "lung_split_provenance.txt")

# The official ICBHI split places these 2 patients in both train and test; force all
# their recordings to the train side.
OVERLAP_PATIENTS = {"156", "218"}

# CinC 2016 training databases; the heart split is computed within these only.
HEART_DBS = {"a", "b", "c", "d", "e"}


def assert_no_patient_leakage(train_ids, test_ids):
    """Raise AssertionError on any patient overlap; log counts on disjoint sets.

    Called at the start of every experiment script. On disjoint sets it prints the
    ``[leakage-check OK] ...`` line and returns ``None``.
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


def make_heart_splits(
    manifest_csv=DEFAULT_MANIFEST,
    out_csv=HEART_SPLITS_CSV,
    test_size=0.20,
    seed=42,
):
    """Build the seeded GroupShuffleSplit heart split within databases A-E.

    Writes a de-duplicated patient-level CSV (patient_id, db_source, split) and asserts zero
    patient leakage on the saved split. Deterministic at seed=42.
    """
    import pandas as pd
    from sklearn.model_selection import GroupShuffleSplit

    df = pd.read_csv(manifest_csv)
    df = df[df.modality == "heart"].copy()
    df["db_source"] = df["db_source"].astype(str)
    # Compute the split within A-E only; the private test set is never re-split.
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


def make_lung_splits(
    manifest_csv=DEFAULT_MANIFEST,
    out_csv=LUNG_SPLITS_CSV,
    provenance_path=LUNG_PROVENANCE,
    test_size=0.40,
    seed=42,
):
    """Build the lung split: official ICBHI (validated) + 156/218 fix, else fallback.

    Official path: adopt the official recording assignments, derive patient_id from the
    recording stem, then force every recording of patients 156 and 218 onto the train side.
    Fallback path: seeded patient-level ``GroupShuffleSplit(test_size=0.40, random_state=42)``.
    Either way, assert disjointness and write the CSV (patient_id, split) plus a provenance
    line recording which path was taken.
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
        # Force all recordings of the overlap patients (156, 218) to train so each patient
        # lands on exactly one side and the disjoint assertion passes.
        repair_mask = out["patient_id"].isin(OVERLAP_PATIENTS)
        out.loc[repair_mask, "split"] = "train"
        out = out.drop_duplicates(subset=["patient_id"])

        provenance = (
            "official-split + 2-patient repair (156,218->train)\n"
            f"source: {source_url}\n"
            "canonical: Harvard Dataverse DOI 10.7910/DVN/HT6PKI\n"
            "2 patients (156, 218) reassigned to train for strict patient-level "
            "integrity; deviates from the official split by 2 patients.\n"
        )
    else:
        # Fallback: seeded patient-level GroupShuffleSplit 60/40.
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
            "patient-level GroupShuffleSplit.\n"
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
