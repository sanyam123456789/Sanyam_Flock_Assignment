"""Download the two REFIT archives, extract them into DATA/raw and DATA/clean, and check
that every file the pipeline expects is there.

    python get_data.py            # both archives (about 15 GB extracted)
    python get_data.py --clean    # cleaned files only (enough for Sections 2-3)

Extraction needs 7-Zip (`7z` on PATH) or the py7zr package (pip install py7zr).
Already-downloaded archives and already-extracted files are reused.
"""
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

from src.config import DATA_DIR, HOUSES_ALL, clean_file, raw_files

ARCHIVES = {
    "clean": ("https://pureportal.strath.ac.uk/files/62090184/CLEAN_REFIT_081116.7z", DATA_DIR / "clean"),
    "raw": ("https://pureportal.strath.ac.uk/files/62090177/REFIT_RAW_081116.7z", DATA_DIR / "raw"),
}


def expected(which: str) -> list[Path]:
    return [clean_file(h) for h in HOUSES_ALL] if which == "clean" else [f for h in HOUSES_ALL for f in raw_files(h)]


def extract(archive: Path, dest: Path) -> None:
    tmp = dest / "_extract"
    tmp.mkdir(parents=True, exist_ok=True)
    if shutil.which("7z"):
        subprocess.run(["7z", "x", "-y", f"-o{tmp}", str(archive)], check=True)
    else:
        try:
            import py7zr
        except ImportError:
            sys.exit("Need 7-Zip (7z on PATH) or `pip install py7zr` to extract " + archive.name)
        with py7zr.SevenZipFile(archive) as z:
            z.extractall(tmp)
    for f in tmp.rglob("*.csv"):  # flatten: CSVs must sit directly in dest
        shutil.move(str(f), dest / f.name)
    shutil.rmtree(tmp)


def main() -> None:
    which = ["clean"] if "--clean" in sys.argv else ["clean", "raw"]
    for w in which:
        url, dest = ARCHIVES[w]
        dest.mkdir(parents=True, exist_ok=True)
        if all(f.exists() for f in expected(w)):
            print(f"{w}: all files already present in {dest}")
            continue
        archive = DATA_DIR / url.rsplit("/", 1)[1]
        if not archive.exists():
            print(f"{w}: downloading {url} (several GB, this takes a while) ...", flush=True)
            urllib.request.urlretrieve(url, archive.with_suffix(".part"))
            archive.with_suffix(".part").rename(archive)
        print(f"{w}: extracting {archive.name} -> {dest}", flush=True)
        extract(archive, dest)
        missing = [f.name for f in expected(w) if not f.exists()]
        if missing:
            sys.exit(f"{w}: still missing after extraction: {', '.join(missing)}")
        print(f"{w}: OK, {len(expected(w))} files. You can delete {archive.name} now.")


if __name__ == "__main__":
    main()
