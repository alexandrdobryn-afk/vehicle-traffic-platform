"""Populate conservative vehicle makes from persisted best vehicle crops."""

import argparse
import asyncio
from pathlib import Path

import cv2
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import sessionmaker

from app.models.database import VehicleTrack, engine
from app.models.model_registry import model_registry
from app.services.brand_recognition_service import BrandRecognitionService


async def reclassify(camera_id: int | None, dry_run: bool) -> None:
    service = BrandRecognitionService(model_registry.get_brand_classifier())
    if service.model is None:
        raise RuntimeError("Brand classifier is unavailable")
    session_factory = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    changed = 0
    skipped = 0
    async with session_factory() as session:
        query = select(VehicleTrack).where(VehicleTrack.best_vehicle_crop_path.isnot(None))
        if camera_id is not None:
            query = query.where(VehicleTrack.camera_id == camera_id)
        tracks = (await session.execute(query.order_by(VehicleTrack.id))).scalars().all()
        for track in tracks:
            image = cv2.imread(str(Path(track.best_vehicle_crop_path)))
            if image is None or image.size == 0:
                skipped += 1
                continue
            observation = service.recognize_details(image)
            # Historical tracks have one saved crop, so require the stronger
            # single-frame threshold used by resolve_brand.
            result = service.resolve_brand([observation])
            make = result["brand"]
            if make == "unknown":
                skipped += 1
                continue
            diagnostics = dict(track.recognition_diagnostics or {})
            diagnostics["brand"] = {
                "status": "recognized",
                "value": make,
                "candidate_value": result.get("candidate_brand"),
                "confidence": round(float(result["confidence"]), 3),
                "engine": "brandeye_logo_detector",
                "support_count": result["support_count"],
                "sample_count": result["sample_count"],
                "distribution": result["distribution"],
                "provisional": True,
            }
            if not dry_run:
                track.vehicle_make = make
                track.make_confidence = float(result["confidence"])
                track.recognition_diagnostics = diagnostics
            changed += 1
            print(f"track={track.id} camera={track.camera_id} make={make} confidence={result['confidence']:.3f}")
        if not dry_run:
            await session.commit()
    print(f"changed={changed} skipped={skipped} dry_run={dry_run}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera-id", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    asyncio.run(reclassify(args.camera_id, args.dry_run))


if __name__ == "__main__":
    main()
