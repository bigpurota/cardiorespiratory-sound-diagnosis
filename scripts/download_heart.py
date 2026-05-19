"""
Download the PhysioNet/CinC 2016 heart-sound training set (databases A-E) into
data/raw/cinc2016/. Expects 3,126 WAV files across training-a..training-e.
"""
import pathlib
import subprocess
import zipfile
import sys
import csv

DEST = pathlib.Path("data/raw/cinc2016")
DEST.mkdir(parents=True, exist_ok=True)

URL = "https://physionet.org/files/challenge-2016/1.0.0/training.zip"
ZIP_PATH = DEST / "training.zip"


def is_valid_zip(path: pathlib.Path) -> bool:
    """Return True if path is a complete, valid ZIP file."""
    try:
        with zipfile.ZipFile(path) as zf:
            bad = zf.testzip()
            return bad is None
    except Exception:
        return False


def download():
    if ZIP_PATH.exists():
        if is_valid_zip(ZIP_PATH):
            print(f"[INFO] {ZIP_PATH} already exists and is valid — skipping download.")
            return
        else:
            print(f"[WARN] {ZIP_PATH} exists but is incomplete/corrupt — re-downloading.")
            ZIP_PATH.unlink()
    print(f"[INFO] Downloading training.zip from PhysioNet (~181 MB)...")
    # Prefer wget; fall back to curl if it isn't installed.
    try:
        subprocess.run(
            ["wget", "-N", "-c", URL, "-O", str(ZIP_PATH)],
            check=True,
        )
    except FileNotFoundError:
        print("[INFO] wget not found, falling back to curl...")
        subprocess.run(
            ["curl", "-L", "-C", "-", URL, "-o", str(ZIP_PATH)],
            check=True,
        )


def extract():
    print(f"[INFO] Extracting {ZIP_PATH} to {DEST} ...")
    with zipfile.ZipFile(ZIP_PATH) as zf:
        zf.extractall(DEST)
    ZIP_PATH.unlink()
    print("[INFO] Extraction complete; training.zip removed.")

    # training-f (114 recordings) was added after the challenge closed and is not
    # part of the A-E set, so drop it to get the canonical 3,126-recording dataset.
    training_f = DEST / "training-f"
    if training_f.exists():
        import shutil
        shutil.rmtree(training_f)
        print("[INFO] Removed training-f/ (post-challenge addition, excluded from A-E set).")


def verify():
    wav_count = sum(1 for _ in DEST.rglob("*.wav"))
    print(f"[INFO] Downloaded {wav_count} WAV files (expected 3126)")
    if wav_count != 3126:
        print(f"[ERROR] Expected 3126 WAV files, got {wav_count}.", file=sys.stderr)
        sys.exit(1)

    if (DEST / "training-f").exists():
        print("[ERROR] training-f/ directory found — it must NOT be present.", file=sys.stderr)
        sys.exit(1)
    print("[INFO] Confirmed: training-f/ is absent.")


def verify_references():
    expected = dict(a=409, b=490, c=31, d=55, e=2141)
    total = 0
    for db_letter, count in expected.items():
        ref = DEST / f"training-{db_letter}" / "REFERENCE.csv"
        if not ref.exists():
            print(f"[ERROR] Missing {ref}", file=sys.stderr)
            sys.exit(1)
        with open(ref, newline="") as fh:
            rows = list(csv.reader(fh))
        actual = len(rows)
        print(f"[INFO] training-{db_letter}: {actual} rows in REFERENCE.csv (expected {count})")
        if actual != count:
            print(
                f"[ERROR] training-{db_letter}: expected {count} rows, got {actual}.",
                file=sys.stderr,
            )
            sys.exit(1)
        total += actual
    print(f"[INFO] REFERENCE.csv total: {total} recordings across A-E.")


def main():
    # If WAVs are already present and training-f is gone, there's nothing to do.
    wav_count = sum(1 for _ in DEST.rglob("*.wav"))
    if wav_count > 0 and not (DEST / "training-f").exists():
        print(f"[INFO] {wav_count} WAV files already present and training-f absent — skipping download/extraction.")
        if ZIP_PATH.exists():
            ZIP_PATH.unlink()
    else:
        download()
        wav_count = sum(1 for _ in DEST.rglob("*.wav"))
        if wav_count == 0:
            extract()
        else:
            print(f"[INFO] {wav_count} WAV files already present — skipping extraction.")
            if ZIP_PATH.exists():
                ZIP_PATH.unlink()
            # A previous interrupted run may have left training-f behind.
            training_f = DEST / "training-f"
            if training_f.exists():
                import shutil
                shutil.rmtree(training_f)
                print("[INFO] Removed training-f/ (post-challenge addition, excluded from A-E set).")

    verify()
    verify_references()
    print("Heart download complete. 3,126 recordings verified.")
    sys.exit(0)


if __name__ == "__main__":
    main()
