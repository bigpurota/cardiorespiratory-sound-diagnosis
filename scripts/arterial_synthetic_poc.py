"""Synthetic arterial-bruit proof-of-concept for the unified pipeline.

IMPORTANT FRAMING. No clinical data are used or claimed. This script generates
*synthetic* neck-auscultation signals — a "bruit" class (systolic-gated turbulent
noise in the carotid-bruit band, after the Lees-Dewey phonoangiography model cited
in Chapter 4) and a "normal" class (low-frequency cardiac thump plus a noise floor,
without the systolic high-frequency component) — and runs the *exact same* pipeline
functions used for heart and lung sounds (Butterworth bandpass -> peak-normalise ->
fixed 3 s windows -> MFCC+delta+spectral feature set B) under a patient-level split.

Its only purpose is to demonstrate, as a runnable complement to the analytical
arterial sub-study, that the pipeline applies end-to-end to arterial-band audio with
nothing more than a change of passband. The reported accuracy is on synthetic signals
and is NOT evidence of real-world bruit detection. Deterministic (seed 42), CPU only.
"""
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GroupShuffleSplit

from src import config
from src.preprocess import bandpass_sos, peak_normalize
from src.segment import segment_fixed
from src.features import window_feature_vector
from src.metrics import majority_vote, heart_macc, save_cm
from src.split import assert_no_patient_leakage

SR = 4000
ARTERIAL_BAND = (50, 800)          # carotid-bruit band (Chapter 4)
N_PER_CLASS = 60                   # synthetic subjects per class
RECS_PER_SUBJECT = 4
REC_SECONDS = 10.0
SEED = 42
TAB = os.path.join(config.RESULTS_DIR, "tables")
FIG = os.path.join(config.RESULTS_DIR, "figures")


def _cardiac_envelope(n, fs, bpm, rng, systolic_frac=0.33):
    """Per-cycle envelope; the systolic third is marked for bruit gating."""
    t = np.arange(n) / fs
    period = 60.0 / bpm
    phase = (t % period) / period
    # low-frequency 'lub-dub' thump present in both classes (<~100 Hz energy)
    thump = np.zeros(n)
    for centre, amp in ((0.12, 1.0), (0.45, 0.7)):  # S1, S2 positions in the cycle
        thump += amp * np.exp(-((phase - centre) ** 2) / (2 * 0.02 ** 2))
    systole = (phase >= 0.08) & (phase <= 0.08 + systolic_frac)
    return thump, systole.astype(float)


def _make_signal(label, rng):
    """label 1 = bruit, 0 = normal. The two classes are made deliberately hard to
    separate: BOTH carry mid-band energy with overlapping amplitude ranges, so the
    only systematic cue is that the bruit's energy is *systolic-gated and structured*
    while the normal's mid-band (benign flow / muscle / artefact) is *diffuse* and on
    average weaker. MFCC summary statistics see the gating only weakly, so the task is
    learnable but far from trivial."""
    n = int(REC_SECONDS * SR)
    bpm = rng.uniform(55, 95)
    thump, systole = _cardiac_envelope(n, SR, bpm, rng)
    low = bandpass_sos(rng.standard_normal(n), 20, 120, fs=SR) * thump
    gain = rng.uniform(0.7, 1.3)                       # per-recording level variation
    baseline = bandpass_sos(rng.standard_normal(n), 1, 8, fs=SR) * 0.15  # slow wander
    noise_floor = rng.uniform(0.06, 0.12) * rng.standard_normal(n)       # strong floor
    sig = 0.6 * low + baseline + noise_floor

    # BOTH classes carry a systolic-gated mid-band component of comparable strength;
    # the only systematic difference is the centre frequency, which is higher for a
    # stenotic bruit (Lees-Dewey) but with class distributions that OVERLAP, so the
    # cue is a subtle spectral-centroid shift buried in noise rather than presence vs
    # absence of energy.
    fc = rng.uniform(330, 600) if label == 1 else rng.uniform(230, 430)
    bw = rng.uniform(60, 120)
    grade = rng.uniform(0.12, 0.38)                    # same grade range for both classes
    mid = bandpass_sos(rng.standard_normal(n), max(fc - bw, 60), fc + bw, fs=SR)
    gate = np.convolve(systole, np.hanning(int(0.04 * SR)), mode="same")
    gate /= max(gate.max(), 1e-9)
    sig = sig + grade * mid * gate
    return (gain * sig).astype("float32")


def _build_dataset():
    rng = np.random.default_rng(SEED)
    rows, X = [], []
    rid = 0
    for label in (0, 1):
        for subj in range(N_PER_CLASS):
            pid = f"{'br' if label else 'nm'}{subj:03d}"
            for _ in range(RECS_PER_SUBJECT):
                rid += 1
                y = _make_signal(label, rng)
                yb = peak_normalize(bandpass_sos(y, *ARTERIAL_BAND, fs=SR))
                for w in segment_fixed(yb, win_s=3.0, hop_s=1.5, fs=SR):
                    X.append(window_feature_vector(w, sr=SR, include_spectral=True))
                    rows.append(dict(recording_id=f"r{rid}", patient_id=pid, label=label))
    meta = pd.DataFrame(rows)
    return np.asarray(X, dtype="float64"), meta


def run():
    os.makedirs(FIG, exist_ok=True)
    X, meta = _build_dataset()
    print(f"synthetic set: {len(meta)} windows, {meta.patient_id.nunique()} subjects, "
          f"{meta.recording_id.nunique()} recordings, dim={X.shape[1]}")

    gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=SEED)
    tr_idx, te_idx = next(gss.split(X, meta.label, groups=meta.patient_id))
    pid = meta.patient_id.to_numpy()
    assert_no_patient_leakage(pid[tr_idx], pid[te_idx])
    yb = meta.label.to_numpy()
    rec = meta.recording_id.to_numpy()

    models = {
        "logreg": LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED),
        "rf": RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                     random_state=SEED, n_jobs=1),
        "svm_rbf": SVC(kernel="rbf", class_weight="balanced", random_state=SEED),
    }
    rows = []
    best = None
    for name, mdl in models.items():
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", mdl)])
        pipe.fit(X[tr_idx], yb[tr_idx])
        pred = pipe.predict(X[te_idx])
        pr = majority_vote(pred, rec[te_idx])
        tr_rec = majority_vote(yb[te_idx], rec[te_idx]).reindex(pr.index)
        m = heart_macc(tr_rec.to_numpy().astype(int), pr.to_numpy().astype(int))
        rows.append(dict(model=name, MAcc=round(m["MAcc"], 4),
                         Se=round(m["Se"], 4), Sp=round(m["Sp"], 4),
                         n_test_rec=int(len(pr))))
        print(f"[{name}] MAcc={m['MAcc']:.4f}  Se={m['Se']:.3f}  Sp={m['Sp']:.3f}")
        if best is None or m["MAcc"] > best[1]:
            best = (name, m["MAcc"], tr_rec.to_numpy().astype(int), pr.to_numpy().astype(int))

    pd.DataFrame(rows).to_csv(os.path.join(TAB, "arterial_synthetic.csv"), index=False)
    save_cm(best[2], best[3], [0, 1],
            f"synthetic arterial PoC ({best[0]}, recording-level)",
            os.path.join(FIG, "cm_arterial_synth.png"))
    print(f"\nbest: {best[0]} MAcc={best[1]:.4f}")
    print(f"wrote {TAB}/arterial_synthetic.csv and {FIG}/cm_arterial_synth.png")


if __name__ == "__main__":
    run()
