"""
Download the ICBHI 2017 respiratory sound database via the Kaggle API into
data/raw/icbhi2017/. Requires ~/.kaggle/kaggle.json (chmod 600). Expects
920 WAV files plus 920 annotation TXT files.
"""
import pathlib
import subprocess
import zipfile
import sys
import os
import stat

DEST = pathlib.Path("data/raw/icbhi2017")
DEST.mkdir(parents=True, exist_ok=True)

KAGGLE_SLUG = "vbookshelf/respiratory-sound-database"
KAGGLE_SLUG_FALLBACK = "nimalanparameshwaran/icbhi-2017-challenge-respiratory-sound-database"


def check_credentials():
    """Abort with setup instructions if the Kaggle token is missing or insecure."""
    kaggle_json = pathlib.Path.home() / ".kaggle" / "kaggle.json"

    if not kaggle_json.exists():
        print(
            "ERROR: ~/.kaggle/kaggle.json not found.",
            file=sys.stderr,
        )
        print(
            "To fix: go to https://www.kaggle.com/settings, click 'Create New Token',",
            file=sys.stderr,
        )
        print(
            "then: mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ "
            "&& chmod 600 ~/.kaggle/kaggle.json",
            file=sys.stderr,
        )
        sys.exit(1)

    # The token must not be group- or world-accessible.
    mode = os.stat(kaggle_json).st_mode
    if (mode & 0o077) != 0:
        print(
            "WARNING: ~/.kaggle/kaggle.json has loose permissions. Fixing to 0o600...",
            file=sys.stderr,
        )
        os.chmod(kaggle_json, 0o600)
        print("[INFO] Permissions fixed: ~/.kaggle/kaggle.json is now 0o600.")

    print("[INFO] Kaggle credentials OK.")


def download():
    """Download the ICBHI dataset via the kaggle CLI; try a fallback slug on failure."""
    existing_wavs = list(DEST.rglob("*.wav"))
    if len(existing_wavs) >= 920:
        print(f"[INFO] {len(existing_wavs)} WAV files already present — skipping download.")
        return

    print(
        f"[INFO] Downloading ICBHI 2017 via Kaggle API (~1.5 GB, this will take a while)..."
    )
    try:
        subprocess.run(
            [
                "uv",
                "run",
                "kaggle",
                "datasets",
                "download",
                "-d",
                KAGGLE_SLUG,
                "--path",
                str(DEST),
            ],
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        print(
            f"[WARN] Primary slug '{KAGGLE_SLUG}' failed (exit code {exc.returncode}). "
            f"Trying fallback slug '{KAGGLE_SLUG_FALLBACK}'...",
            file=sys.stderr,
        )
        subprocess.run(
            [
                "uv",
                "run",
                "kaggle",
                "datasets",
                "download",
                "-d",
                KAGGLE_SLUG_FALLBACK,
                "--path",
                str(DEST),
            ],
            check=True,
        )


def extract():
    """Unzip every .zip under DEST and remove the archive afterwards."""
    for zip_path in list(DEST.rglob("*.zip")):
        print(f"[INFO] Extracting {zip_path} ...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(DEST)
        zip_path.unlink()
        print(f"[INFO] {zip_path.name} extracted and removed.")


def verify():
    """Check that exactly 920 WAVs and 920 matching annotation TXTs are present."""
    wav_files = list(DEST.rglob("*.wav"))
    wav_count = len(wav_files)
    print(f"[INFO] WAV files found: {wav_count} (expected 920)")
    if wav_count != 920:
        print(
            f"[ERROR] Expected 920 WAV files, got {wav_count}.", file=sys.stderr
        )
        sys.exit(1)

    wav_stems = {p.stem for p in wav_files}
    txt_stems = {p.stem for p in DEST.rglob("*.txt") if p.stem in wav_stems}
    txt_count = len(txt_stems)
    print(f"[INFO] Annotation TXT files found: {txt_count} (expected 920)")
    if txt_count != 920:
        print(
            f"[ERROR] Expected 920 annotation TXT files, got {txt_count}.", file=sys.stderr
        )
        sys.exit(1)


def discover_split_file():
    """Locate and log any non-annotation TXT files (e.g. split/metadata files)."""
    wav_files = list(DEST.rglob("*.wav"))
    wav_stems = {p.stem for p in wav_files}
    all_txt = list(DEST.rglob("*.txt"))
    non_annotation_txt = [f for f in all_txt if f.stem not in wav_stems]

    if non_annotation_txt:
        print("[INFO] Split/metadata TXT files found:")
        for f in sorted(non_annotation_txt):
            print(f"  {f}")
    else:
        print("[WARN] No split/metadata TXT files found — the split may need to be reconstructed.")

    return non_annotation_txt


def main():
    check_credentials()
    download()
    extract()
    verify()
    split_files = discover_split_file()
    print("Lung download complete. 920 WAVs and 920 annotation TXTs verified.")
    if split_files:
        print(f"[INFO] {len(split_files)} split/metadata file(s) discovered.")
    sys.exit(0)


if __name__ == "__main__":
    main()
