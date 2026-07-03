"""License-plate text normalization and regional validation profiles."""

from __future__ import annotations

import re
from typing import Dict, List, Tuple


# Ukrainian plates use glyphs shared by the Latin and Cyrillic alphabets.
UA_LATIN_LETTERS = "ABCEHIKMOPTX"
CYRILLIC_TO_LATIN = str.maketrans(
    {
        "А": "A",
        "В": "B",
        "С": "C",
        "Е": "E",
        "Н": "H",
        "І": "I",
        "К": "K",
        "М": "M",
        "О": "O",
        "Р": "P",
        "Т": "T",
        "Х": "X",
    }
)

DIGIT_CONFUSIONS = str.maketrans({"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "Z": "2", "S": "5", "G": "6", "B": "8"})
LETTER_CONFUSIONS = str.maketrans({"0": "O", "1": "I", "2": "Z", "4": "A", "5": "S", "6": "G", "8": "B"})


PLATE_REGEX_PROFILES: Dict[str, List[re.Pattern[str]]] = {
    "UA": [
        re.compile(rf"^[{UA_LATIN_LETTERS}]{{2}}\d{{4}}[{UA_LATIN_LETTERS}]{{2}}$"),
    ],
    "EU": [
        re.compile(r"^[A-Z]{1,3}\d{1,4}[A-Z]{0,3}$"),
        re.compile(r"^[A-Z]{1,3}-[A-Z]{1,3}\d{1,4}$"),
    ],
    "UK": [
        re.compile(r"^[A-Z]{2}\d{2}[A-Z]{3}$"),
        re.compile(r"^[A-Z]\d{1,3}[A-Z]{3}$"),
        re.compile(r"^[A-Z]{3}\d{1,3}[A-Z]$"),
    ],
    # Indian private/commercial registrations, including legacy one-letter
    # series. Examples: KA02MM9091, DL1CAB1234, MH12AB1234.
    "IN": [
        re.compile(r"^[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{4}$"),
    ],
    "US": [
        re.compile(r"^[A-Z0-9]{2,8}$"),
    ],
    "CUSTOM": [],
}


def normalize_plate_text(text: str) -> str:
    """Normalize alphabet confusables and remove separators/noise."""
    normalized = (text or "").upper().strip().translate(CYRILLIC_TO_LATIN)
    return re.sub(r"[^A-Z0-9]", "", normalized)


def normalize_plate_for_profile(text: str, profile: str = "AUTO") -> str:
    """Apply safe positional glyph corrections for structured plate profiles."""
    normalized = normalize_plate_text(text)
    profile_name = profile.upper()
    if profile_name == "AUTO" and any(
        pattern.fullmatch(normalized)
        for name in ("UA", "UK", "IN", "EU")
        for pattern in PLATE_REGEX_PROFILES[name]
    ):
        return normalized
    if profile_name in ("AUTO", "IN") and 8 <= len(normalized) <= 11:
        # IN: state letters + 1/2 district digits + 1..3 series letters
        # + four registration digits. Try the finite set of legal splits and
        # repair common O/0, I/1, S/5 OCR substitutions positionally.
        for district_len in (2, 1):
            for series_len in (3, 2, 1):
                if 2 + district_len + series_len + 4 != len(normalized):
                    continue
                district_end = 2 + district_len
                series_end = district_end + series_len
                candidate = (
                    normalized[:2].translate(LETTER_CONFUSIONS)
                    + normalized[2:district_end].translate(DIGIT_CONFUSIONS)
                    + normalized[district_end:series_end].translate(LETTER_CONFUSIONS)
                    + normalized[series_end:].translate(DIGIT_CONFUSIONS)
                )
                if any(pattern.fullmatch(candidate) for pattern in PLATE_REGEX_PROFILES["IN"]):
                    return candidate
    if profile_name not in ("AUTO", "UK") or not (5 <= len(normalized) <= 7):
        return normalized
    chars = list(normalized)
    for index in range(min(len(chars), 2)):
        chars[index] = chars[index].translate(LETTER_CONFUSIONS)
    for index in range(2, min(len(chars), 4)):
        chars[index] = chars[index].translate(DIGIT_CONFUSIONS)
    for index in range(4, len(chars)):
        chars[index] = chars[index].translate(LETTER_CONFUSIONS)
    return "".join(chars)


def validate_plate(text: str, profile: str = "UA") -> Tuple[bool, float]:
    """Validate a normalized plate and return validity plus a soft score."""
    normalized = normalize_plate_text(text)
    profile_name = profile.upper()
    if profile_name == "AUTO":
        for name, score in (("UK", 1.0), ("UA", 1.0), ("IN", 1.0), ("EU", 0.85)):
            if any(pattern.fullmatch(normalized) for pattern in PLATE_REGEX_PROFILES[name]):
                return True, score
        patterns = []
    else:
        patterns = PLATE_REGEX_PROFILES.get(profile_name, [])
    if any(pattern.fullmatch(normalized) for pattern in patterns):
        return True, 1.0

    # Near matches remain candidates for temporal voting, but cannot be verified.
    if 6 <= len(normalized) <= 10 and re.search(r"\d{2,}", normalized) and re.search(
        r"[A-Z]", normalized
    ):
        return False, 0.35
    return False, 0.0
