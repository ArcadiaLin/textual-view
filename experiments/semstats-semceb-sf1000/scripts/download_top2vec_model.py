"""Download pinned sentence-transformers/all-MiniLM-L6-v2 locally for SemStats offline Top2Vec."""
import hashlib
import json
from pathlib import Path

from huggingface_hub import snapshot_download

EXP_DIR = Path(__file__).resolve().parent.parent
REPO_ID = "sentence-transformers/all-MiniLM-L6-v2"
REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
TARGET_DIR = EXP_DIR / "models" / "all-MiniLM-L6-v2"
MANIFEST_PATH = EXP_DIR / "models" / "all-MiniLM-L6-v2.manifest.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def main() -> None:
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id=REPO_ID,
        revision=REVISION,
        local_dir=str(TARGET_DIR),
    )
    weights = TARGET_DIR / "model.safetensors"
    manifest = {
        "repo_id": REPO_ID,
        "revision": REVISION,
        "local_dir": str(TARGET_DIR),
        "model_safetensors_sha256": sha256_file(weights) if weights.exists() else None,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Downloaded {REPO_ID}@{REVISION} to {path}")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
