"""Leakage-safe patient-level train/test splits."""
import os
import sys

from src import config

sys.path.insert(0, config.PROJECT_ROOT)

DEFAULT_MANIFEST = os.path.join(config.DATA_PROCESSED, "manifest.csv")
HEART_SPLITS_CSV = os.path.join(config.SPLITS_DIR, "heart_splits.csv")
LUNG_SPLITS_CSV = os.path.join(config.SPLITS_DIR, "lung_splits.csv")
LUNG_PROVENANCE = os.path.join(config.SPLITS_DIR, "lung_split_provenance.txt")

OVERLAP_PATIENTS = {"156", "218"}

HEART_DBS = {"a", "b", "c", "d", "e"}


def assert_no_patient_leakage(train_ids, test_ids):
    """Raise AssertionError on any patient overlap; log counts on"""
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
    """Build the seeded GroupShuffleSplit heart split within"""
    import pandas as pd
    from sklearn.model_selection import GroupShuffleSplit

    df = pd.read_csv(manifest_csv)
    df = df[df.modality == "heart"].copy()
    df["db_source"] = df["db_source"].astype(str)
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
    """Build the lung split: official ICBHI (validated) + 156/218"""
    import pandas as pd

    from scripts.fetch_icbhi_split import fetch_official_split

    manifest = pd.read_csv(manifest_csv)
    lung = manifest[manifest.modality == "lung"].copy()
    lung["patient_id"] = lung["patient_id"].astype(str)

    rows, source_url = fetch_official_split()

    if rows is not None:
        rec = pd.DataFrame(rows, columns=["stem", "split"])
        rec["patient_id"] = rec["stem"].str.split("_").str[0].astype(str)
        rec["split"] = rec["split"].astype(str)

        out = rec[["patient_id", "split"]].copy()
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
