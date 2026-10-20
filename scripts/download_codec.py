"""Download pinned TAESD image-codec weights and verify SHA256."""

import argparse
import hashlib
import json
from pathlib import Path
import urllib.request


REVISION = "d968ebf70e930b76243c773a63162ed94482361f"
BASE = f"https://raw.githubusercontent.com/madebyollin/taesd/{REVISION}"
FILES = {
    "taesd_encoder.pth": "15bc6128f0ac51c673d3427216082ebfa62402ffc89763af74edfe259b32d49d",
    "taesd_decoder.pth": "02873377c3f4659cd9f9adb2f718dcb434ba4c7bba3af3ee7bb95cbdabe2d3cf",
    "LICENSE": "261497cf8e9ed613ed1fa5ddf48af32cdf50545b2ccd9edd7eacc91fb9255d6e",
}


def download_file(url, path, sha256, verify_only=False):
    path = Path(path)
    if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == sha256:
        return
    if verify_only:
        raise ValueError(f"SHA256 verification failed: {path}")
    with urllib.request.urlopen(url, timeout=60) as response:
        data = response.read()
    if hashlib.sha256(data).hexdigest() != sha256:
        raise ValueError(f"SHA256 mismatch from {url}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    temporary.write_bytes(data)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/raw/taesd"))
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    records = {}
    for name, checksum in FILES.items():
        url, path = f"{BASE}/{name}", args.output / name
        download_file(url, path, checksum, args.verify_only)
        records[name] = {"source": url, "sha256": checksum, "bytes": path.stat().st_size}
        print(f"Verified {name}", flush=True)
    if not args.verify_only:
        (args.output / "manifest.json").write_text(json.dumps({"revision": REVISION, "files": records}, indent=2) + "\n")


if __name__ == "__main__":
    main()
