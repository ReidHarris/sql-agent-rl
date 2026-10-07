import zipfile
from pathlib import Path

ZIP_PATH = Path("spider_data.zip")
DEST = Path("data")
ILLEGAL = set('<>:"|?*')

skipped = 0
with zipfile.ZipFile(ZIP_PATH) as z:
    for info in z.infolist():
        name = info.filename
        if name.startswith("__MACOSX") or any(c in ILLEGAL for c in name):
            skipped += 1
            continue
        z.extract(info, DEST)

print(f"Done. Skipped {skipped} entries (macOS metadata or names invalid on Windows).")