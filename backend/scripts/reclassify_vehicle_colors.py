"""Reclassify persisted vehicle crops with the current color pipeline."""

import argparse
import asyncio
from pathlib import Path

import cv2
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import sessionmaker

from app.models.database import VehicleTrack, engine
from app.services.color_recognition_service import ColorRecognitionService


async def reclassify(camera_id: int | None, dry_run: bool) -> None:
    service = ColorRecognitionService()
    session_factory = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    changed = 0
    skipped = 0
    async with session_factory() as session:
        query = select(VehicleTrack).where(VehicleTrack.best_vehicle_crop_path.isnot(None))
        if camera_id is not None:
            query = query.where(VehicleTrack.camera_id == camera_id)
        tracks = (await session.execute(query.order_by(VehicleTrack.id))).scalars().all()
        for track in tracks:
            path = Path(track.best_vehicle_crop_path)
            image = cv2.imread(str(path))
            if image is None or image.size == 0:
                skipped += 1
                continue
            result = service.recognize_details(image)
            color = result["color"]
            confidence = float(result["confidence"])
            if color == "unknown" or confidence < 0.35:
                skipped += 1
                continue
            previous = track.color
            diagnostics = dict(track.recognition_diagnostics or {})
            diagnostics["color"] = {
                "status": "recognized",
                "value": color,
                "confidence": round(confidence, 3),
                "engine": "hsv_multizone",
                "html_hex": result["hex_code"],
                "candidate_value": color,
                "sample_count": 1,
                "support_count": 1,
                "distribution": {color: 100.0},
                "provisional": True,
                "reclassified_from": previous,
            }
            if not dry_run:
                track.color = color
                track.color_confidence = confidence
                track.recognition_diagnostics = diagnostics
            changed += 1
            print(f"track={track.id} camera={track.camera_id} {previous}->{color} {confidence:.3f}")
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
