"""Probe camera URLs with the same decoder used by the backend."""
import argparse
import json
import sys

from app.services.video_capture_service import VideoCaptureService


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        metavar="TYPE=URL",
        help="Source to probe; TYPE is rtsp, hls, mjpeg or jpeg",
    )
    parser.add_argument("--timeout", type=float, default=12.0)
    args = parser.parse_args()

    results = []
    for source in args.source:
        if "=" not in source:
            parser.error(f"Invalid --source value: {source!r}")
        source_type, url = source.split("=", 1)
        result = VideoCaptureService.test_connection(
            url=url,
            source_type=source_type,
            timeout=args.timeout,
        )
        result["url"] = VideoCaptureService._mask_url(url)
        results.append(result)

    print(json.dumps(results, indent=2, ensure_ascii=True))
    return 0 if all(item.get("success") for item in results) else 1


if __name__ == "__main__":
    sys.exit(main())
