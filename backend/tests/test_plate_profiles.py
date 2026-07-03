import unittest

from app.services.plate_profiles import normalize_plate_for_profile, normalize_plate_text, validate_plate


class PlateProfileTests(unittest.TestCase):
    def test_normalizes_ukrainian_cyrillic_glyphs(self):
        self.assertEqual(normalize_plate_text("АА 1234 ВВ"), "AA1234BB")

    def test_accepts_standard_ua_plate(self):
        self.assertEqual(validate_plate("KA1234AA", "UA"), (True, 1.0))

    def test_rejects_letters_not_used_on_standard_ua_plates(self):
        valid, score = validate_plate("ZZ1234ZZ", "UA")
        self.assertFalse(valid)
        self.assertLess(score, 1.0)

    def test_keeps_near_match_for_temporal_voting(self):
        self.assertEqual(validate_plate("AA123BB", "UA"), (False, 0.35))

    def test_accepts_uk_plate_in_uk_and_auto_profiles(self):
        self.assertEqual(validate_plate("GX15OCJ", "UK"), (True, 1.0))
        self.assertEqual(validate_plate("GX15OCJ", "AUTO"), (True, 1.0))

    def test_auto_still_accepts_ua_plate(self):
        self.assertEqual(validate_plate("KA1234AA", "AUTO"), (True, 1.0))

    def test_accepts_indian_plate_in_india_and_auto_profiles(self):
        self.assertEqual(validate_plate("KA02MM9091", "IN"), (True, 1.0))
        self.assertEqual(validate_plate("KA02MM9091", "AUTO"), (True, 1.0))
        self.assertEqual(validate_plate("DL1CAB1234", "AUTO"), (True, 1.0))
        self.assertEqual(normalize_plate_for_profile("KAO2MM9O91", "AUTO"), "KA02MM9091")
        self.assertEqual(normalize_plate_for_profile("K402HN1826", "AUTO"), "KA02HN1826")

    def test_uk_positional_confusions_are_corrected(self):
        self.assertEqual(normalize_plate_for_profile("GXISOGJ", "AUTO"), "GX15OGJ")
        self.assertEqual(normalize_plate_for_profile("GXI5OC", "UK"), "GX15OC")


if __name__ == "__main__":
    unittest.main()
