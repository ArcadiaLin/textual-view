"""Unit tests for unsigned and SemCEB-native signed q-error evaluation."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import evaluate_semceb as ev


class EvaluateSemcebTests(unittest.TestCase):
    def test_unsigned_qerr_zero_and_symmetry(self):
        self.assertEqual(ev.qerr(0.0, 0.0), 1.0)
        self.assertEqual(ev.qerr(0.5, 0.0), 1.0)
        self.assertEqual(ev.qerr(0.0, 10.0), 10.0)
        self.assertEqual(ev.qerr(10.0, 0.0), 10.0)
        self.assertAlmostEqual(ev.qerr(5.0, 20.0), 4.0)
        self.assertAlmostEqual(ev.qerr(20.0, 5.0), 4.0)

    def test_semceb_signed_qerr_matches_reference(self):
        self.assertEqual(ev.semceb_signed_qerr(0.0, 0.0), (0, 1.0))
        self.assertEqual(ev.semceb_signed_qerr(0.4, 0.0), (0, 1.0))
        self.assertEqual(ev.semceb_signed_qerr(0.0, 1.0), (0, -1.0))
        self.assertEqual(ev.semceb_signed_qerr(0.0, 10.0), (0, -10.0))
        self.assertEqual(ev.semceb_signed_qerr(10.0, 0.0), (10, 10.0))
        self.assertEqual(ev.semceb_signed_qerr(5.0, 20.0), (5, -4.0))
        self.assertEqual(ev.semceb_signed_qerr(20.0, 5.0), (20, 4.0))
        self.assertEqual(ev.semceb_signed_qerr(10.0, 10.0), (10, 1.0))

    def test_summarize_semceb_signed_counts(self):
        rows = [
            {"true": 10.0, "estimate_int": 5, "semceb_signed_qerr": -2.0},
            {"true": 10.0, "estimate_int": 10, "semceb_signed_qerr": 1.0},
            {"true": 10.0, "estimate_int": 30, "semceb_signed_qerr": 3.0},
        ]
        s = ev.summarize_semceb_signed(rows)
        self.assertEqual(s["underestimate_count"], 1)
        self.assertEqual(s["exact_count"], 1)
        self.assertEqual(s["overestimate_count"], 1)
        self.assertAlmostEqual(s["signed_mean"], (-2.0 + 1.0 + 3.0) / 3.0)
        self.assertAlmostEqual(s["signed_min"], -2.0)
        self.assertAlmostEqual(s["signed_max"], 3.0)
        self.assertAlmostEqual(s["abs_p50"], 2.0)


if __name__ == "__main__":
    unittest.main()
