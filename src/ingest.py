"""
src/ingest.py — build the recording-level manifest for both modalities (Phase 2).

Implements 02-RESEARCH.md Code Examples §2 (heart) and §3 (lung). The manifest is
the single source of truth every downstream phase reads (D-05): ONE row per `.wav`
with the exact columns

    filepath, patient_id, label, modality, duration_s, db_source, segment_id

Lung cycle labels (per-respiratory-cycle) are written to a SEPARATE
``lung_cycles.csv`` so the manifest stays one-row-per-recording (Open Question 2
resolution + Pitfall 4). Durations come from ``soundfile.info().duration``
(header-only, fast — no full decode). `wfdb` is NOT needed (labels live in
REFERENCE.csv / the annotation `.txt` files).

`import config` runs first for the SEED=42 determinism side effect.
"""
import config  # noqa: F401 — import FIRST for the SEED=42 side effect (determinism)

import csv
import pathlib

import pandas as pd
import soundfile as sf

from src.config_loader import load_params

__all__ = ["heart_records", "lung_records", "build_manifest"]

# Exact, order-sensitive manifest header (ROADMAP Phase-2 success criterion).
MANIFEST_COLUMNS = [
    "filepath",
    "patient_id",
    "label",
    "modality",
    "duration_s",
    "db_source",
    "segment_id",
]
LUNG_CYCLE_COLUMNS = ["filepath", "patient_id", "cycle_idx", "start_s", "end_s", "label"]

# CinC databases A–E (patient-level CV happens WITHIN these only, D-10).
CINC_DBS = ["training-a", "training-b", "training-c", "training-d", "training-e"]

# 4-class ICBHI cycle label from the two binary annotation flags [crackle, wheeze]
# (D-11): 00 → normal, 10 → crackle, 01 → wheeze, 11 → both.
FLAGS_TO_LABEL = {(0, 0): "normal", (1, 0): "crackle", (0, 1): "wheeze", (1, 1): "both"}

# Default audio directory inside the ICBHI mirror layout.
ICBHI_AUDIO_SUBPATH = pathlib.Path(
    "Respiratory_Sound_Database"
) / "Respiratory_Sound_Database" / "audio_and_txt_files"


def heart_records(cinc_root, sr_target=4000):
    """Build recording-level heart rows from CinC 2016 REFERENCE.csv joins (§2).

    Reads each ``training-{a..e}/REFERENCE.csv`` (format ``record,label`` with
    label ∈ {-1=normal, 1=abnormal}, D-08), asserts the matching ``{stem}.wav``
    exists (total-join assertion), reads its duration via ``sf.info``, and emits a
    dict row with ``patient_id`` = the DB-prefixed stem (D-10), ``db_source`` =
    the DB letter, and ``segment_id`` = None (heart 3-s windows are cut lazily in
    Phase 3, D-06). Logs and asserts the global label distribution (2495×-1 /
    631×1, Pitfall 2) and the unsure(0) exclusion count (= 0, D-09).
    """
    cinc_root = pathlib.Path(cinc_root)
    rows = []
    n_normal = n_abnormal = n_unsure = 0

    for db in CINC_DBS:
        ddir = cinc_root / db
        ref_path = ddir / "REFERENCE.csv"
        with open(ref_path, newline="") as fh:
            ref = {r[0]: int(r[1]) for r in csv.reader(fh) if r}
        for stem, label in ref.items():
            if label == 0:  # "unsure" — excluded from binary targets (D-09)
                n_unsure += 1
                continue
            wav = ddir / f"{stem}.wav"
            assert wav.exists(), f"missing heart wav for REFERENCE entry: {wav}"
            dur = sf.info(wav).duration  # header read, no full decode
            if label == -1:
                n_normal += 1
            elif label == 1:
                n_abnormal += 1
            rows.append(
                dict(
                    filepath=str(wav),
                    patient_id=stem,        # DB-prefixed stem, e.g. a0001 (D-10)
                    label=label,            # -1 = normal, 1 = abnormal (D-08)
                    modality="heart",
                    duration_s=round(dur, 3),
                    db_source=db[-1],       # a..e
                    segment_id=None,        # assigned lazily in Phase 3 (D-06)
                )
            )

    print(
        f"[heart] label distribution: normal(-1)={n_normal} abnormal(1)={n_abnormal} "
        f"(total {n_normal + n_abnormal})"
    )
    print(f"[heart] unsure(0) exclusion count = {n_unsure} (D-09)")
    assert n_normal == 2495 and n_abnormal == 631, (
        f"heart label distribution sanity FAILED (Pitfall 2): expected 2495 normal / "
        f"631 abnormal, got {n_normal} / {n_abnormal} — possible label inversion or "
        f"missing records"
    )
    return rows


def lung_records(audio_dir):
    """Build lung manifest rows + per-cycle rows from ICBHI annotations (§3).

    Emits ONE manifest row per ``.wav`` (920 rows, ``segment_id=None``,
    ``db_source="icbhi"``, ``patient_id`` = leading numeric field of the filename
    per D-11, duration via ``sf.info``) AND a SEPARATE list of cycle rows — one per
    annotation line ``start<TAB>end<TAB>crackle<TAB>wheeze`` — carrying the 4-class
    label from ``FLAGS_TO_LABEL``.

    Returns
    -------
    (manifest_rows, cycle_rows) : tuple[list[dict], list[dict]]
    """
    audio_dir = pathlib.Path(audio_dir)
    manifest_rows = []
    cycle_rows = []

    for wav in sorted(audio_dir.glob("*.wav")):
        patient_id = wav.stem.split("_")[0]  # leading numeric, e.g. "101" (D-11)
        txt = wav.with_suffix(".txt")
        dur = sf.info(wav).duration

        manifest_rows.append(
            dict(
                filepath=str(wav),
                patient_id=patient_id,
                label=None,                 # cycle labels live in lung_cycles.csv
                modality="lung",
                duration_s=round(dur, 3),
                db_source="icbhi",
                segment_id=None,            # one row per recording (D-05/Pitfall 4)
            )
        )

        with open(txt) as fh:
            for idx, line in enumerate(fh):
                if not line.strip():
                    continue
                s, e, cr, wh = line.split()
                label = FLAGS_TO_LABEL[(int(cr), int(wh))]
                cycle_rows.append(
                    dict(
                        filepath=str(wav),
                        patient_id=patient_id,
                        cycle_idx=idx,
                        start_s=round(float(s), 3),
                        end_s=round(float(e), 3),
                        label=label,
                    )
                )

    # 4-class cycle distribution sanity (VERIFIED 3642/1864/886/506).
    dist = pd.Series([r["label"] for r in cycle_rows]).value_counts().to_dict()
    print(
        f"[lung] recordings={len(manifest_rows)} cycles={len(cycle_rows)} "
        f"4-class dist={ {k: int(dist.get(k, 0)) for k in ('normal','crackle','wheeze','both')} }"
    )
    return manifest_rows, cycle_rows


def build_manifest(cinc_root=None, icbhi_audio_dir=None, out_dir=None):
    """Assemble manifest.csv (heart + lung) + lung_cycles.csv.

    Writes ``data/processed/manifest.csv`` with the exact column order and
    ``data/processed/lung_cycles.csv`` with the per-cycle columns. Returns the two
    output paths.
    """
    cinc_root = pathlib.Path(cinc_root or config.CINC2016_DIR)
    if icbhi_audio_dir is None:
        icbhi_audio_dir = pathlib.Path(config.ICBHI2017_DIR) / ICBHI_AUDIO_SUBPATH
    icbhi_audio_dir = pathlib.Path(icbhi_audio_dir)
    out_dir = pathlib.Path(out_dir or config.DATA_PROCESSED)
    out_dir.mkdir(parents=True, exist_ok=True)

    heart_rows = heart_records(cinc_root)
    lung_manifest_rows, lung_cycle_rows = lung_records(icbhi_audio_dir)

    manifest = pd.DataFrame(heart_rows + lung_manifest_rows, columns=MANIFEST_COLUMNS)
    cycles = pd.DataFrame(lung_cycle_rows, columns=LUNG_CYCLE_COLUMNS)

    manifest_path = out_dir / "manifest.csv"
    cycles_path = out_dir / "lung_cycles.csv"
    manifest.to_csv(manifest_path, index=False)
    cycles.to_csv(cycles_path, index=False)

    print(
        f"[manifest] wrote {manifest_path} "
        f"(heart={len(heart_rows)} lung={len(lung_manifest_rows)} rows)"
    )
    print(f"[manifest] wrote {cycles_path} ({len(lung_cycle_rows)} cycle rows)")
    return manifest_path, cycles_path
