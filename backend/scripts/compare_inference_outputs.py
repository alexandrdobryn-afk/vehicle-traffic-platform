"""Fail closed when an accelerated runtime changes recognition outputs."""

import argparse
import json
from pathlib import Path


def _load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _compare_detections(reference, candidate, label, box_tolerance, confidence_tolerance):
    errors = []
    if len(reference) != len(candidate):
        return [f"{label}: count changed {len(reference)} -> {len(candidate)}"]
    for index, (expected, actual) in enumerate(zip(reference, candidate)):
        prefix = f"{label}[{index}]"
        if expected.get("class_id") != actual.get("class_id"):
            errors.append(
                f"{prefix}: class changed {expected.get('class_id')} -> {actual.get('class_id')}"
            )
        if max(abs(float(a) - float(b)) for a, b in zip(expected["bbox"], actual["bbox"])) > box_tolerance:
            errors.append(f"{prefix}: bbox moved beyond {box_tolerance}px")
        if abs(float(expected["confidence"]) - float(actual["confidence"])) > confidence_tolerance:
            errors.append(f"{prefix}: confidence changed beyond {confidence_tolerance}")
    return errors


def compare(reference: dict, candidate: dict) -> list[str]:
    errors = []
    if reference.keys() != candidate.keys():
        errors.append("top-level output groups changed")

    for name, expected in reference.get("vehicle", {}).items():
        actual = candidate.get("vehicle", {}).get(name)
        if actual is None:
            errors.append(f"vehicle/{name}: missing")
            continue
        errors.extend(_compare_detections(expected, actual, f"vehicle/{name}", 2.0, 0.01))

    for name, expected in reference.get("plate", {}).items():
        actual = candidate.get("plate", {}).get(name)
        if actual is None:
            errors.append(f"plate/{name}: missing")
            continue
        errors.extend(_compare_detections(expected, actual, f"plate/{name}", 2.0, 0.02))

    for name, expected in reference.get("ocr", {}).items():
        actual = candidate.get("ocr", {}).get(name)
        if expected is None or actual is None:
            if expected != actual:
                errors.append(f"ocr/{name}: result presence changed")
            continue
        for key in ("text", "raw_text", "regex_valid"):
            if expected.get(key) != actual.get(key):
                errors.append(f"ocr/{name}: {key} changed")
        if abs(float(expected["confidence"]) - float(actual["confidence"])) > 0.05:
            errors.append(f"ocr/{name}: confidence changed beyond 0.05")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference")
    parser.add_argument("candidate")
    args = parser.parse_args()
    errors = compare(_load(args.reference), _load(args.candidate))
    if errors:
        print("REGRESSION FAILED")
        for error in errors:
            print(f"- {error}")
        return 1
    print("REGRESSION PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
