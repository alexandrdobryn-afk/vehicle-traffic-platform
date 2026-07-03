#!/usr/bin/env python3
"""Download approved baseline model artifacts and record their provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "backend" / "models"
MANIFEST_PATH = MODELS_DIR / "manifest.json"
LOCK_PATH = MODELS_DIR / "manifest.lock.json"
MAX_ARTIFACT_BYTES = 500 * 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "VTP-model-manager/1.0"})
    temp_path: Path | None = None
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            content_length = response.headers.get("Content-Length")
            if content_length and int(content_length) > MAX_ARTIFACT_BYTES:
                raise ValueError(f"Artifact is too large: {content_length} bytes")
            with tempfile.NamedTemporaryFile(
                mode="wb", delete=False, dir=destination.parent, suffix=".part"
            ) as temp:
                temp_path = Path(temp.name)
                total = 0
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > MAX_ARTIFACT_BYTES:
                        raise ValueError("Artifact exceeded the configured size limit")
                    temp.write(chunk)
        os.replace(temp_path, destination)
    finally:
        if temp_path and temp_path.exists():
            temp_path.unlink()


def load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Model manifest not found: {MANIFEST_PATH}")
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def selected_artifacts(manifest: dict, names: list[str]) -> list[dict]:
    artifacts = manifest.get("artifacts", [])
    if names == ["all"]:
        return artifacts
    by_name = {artifact["id"]: artifact for artifact in artifacts}
    missing = [name for name in names if name not in by_name]
    if missing:
        raise ValueError(f"Unknown model ids: {', '.join(missing)}")
    return [by_name[name] for name in names]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="*", default=["all"], help="Model ids or 'all'")
    parser.add_argument("--force", action="store_true", help="Redownload existing files")
    parser.add_argument("--verify-only", action="store_true", help="Do not use the network")
    args = parser.parse_args()

    manifest = load_manifest()
    lock = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "artifacts": [],
    }

    print("Model licenses are binding. Ultralytics artifacts require AGPL-3.0 compliance")
    print("or an Ultralytics Enterprise license for closed commercial deployment.\n")

    failed = False
    for artifact in selected_artifacts(manifest, args.models):
        path = MODELS_DIR / artifact["path"]
        try:
            if args.verify_only and not path.exists():
                raise FileNotFoundError(path)
            if not args.verify_only and (args.force or not path.exists()):
                print(f"Downloading {artifact['id']} -> {path}")
                download(artifact["url"], path)
            digest = sha256_file(path)
            size = path.stat().st_size
            if size < 100_000:
                raise ValueError(f"Artifact is unexpectedly small: {size} bytes")
            print(f"OK {artifact['id']}: {size / 1_000_000:.1f} MB sha256={digest[:16]}...")
            lock["artifacts"].append(
                {
                    **artifact,
                    "sha256": digest,
                    "size_bytes": size,
                    "verified_at": datetime.now(timezone.utc).isoformat(),
                }
            )
        except Exception as exc:
            failed = True
            print(f"ERROR {artifact['id']}: {exc}", file=sys.stderr)

    if lock["artifacts"]:
        LOCK_PATH.write_text(json.dumps(lock, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"\nWrote provenance lock: {LOCK_PATH}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
