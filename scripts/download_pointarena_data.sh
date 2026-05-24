#!/usr/bin/env bash
set -euo pipefail

DATASET="${DATASET:-PointArena/pointarena-data}"

python - <<'PY'
try:
    import huggingface_hub  # noqa: F401
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: huggingface_hub. Install it with `pip install huggingface_hub`."
    ) from exc
PY

huggingface-cli download "${DATASET}" \
  --repo-type dataset \
  --local-dir data

python - <<'PY'
from pathlib import Path
import shutil
import zipfile

for name in ("selected_images.zip", "selected_masks.zip"):
    path = Path(name)
    if not path.exists():
        raise SystemExit(f"Expected {name} after download, but it is missing.")
    with zipfile.ZipFile(path) as zf:
        zf.extractall(".")

def promote_dir(src_name, dst_name):
    src = Path(src_name)
    dst = Path(dst_name)
    if not src.exists() or src.resolve() == dst.resolve():
        return
    dst.mkdir(exist_ok=True)
    for child in src.iterdir():
        target = dst / child.name
        if target.exists():
            continue
        shutil.move(str(child), str(target))

promote_dir("selected_images", "images")
promote_dir("selected_masks", "masks")

missing = [p for p in ("data.json", "pixmo_metadata.csv", "images", "masks") if not Path(p).exists()]
if missing:
    raise SystemExit(f"Downloaded data, but these expected paths are still missing: {missing}")

print("PointArena data is ready: data.json, pixmo_metadata.csv, images/, masks/")
PY
