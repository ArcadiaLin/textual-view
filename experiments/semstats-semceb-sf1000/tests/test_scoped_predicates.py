"""Offline checks for evaluation-query scope and source provenance; no LLM calls."""
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import prepare_semceb_sf1000 as prep


class ScopedPredicateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mappings = prep.load_scoped_predicates()
        cls.sources = {
            f"Q{q['id']:03d}": q
            for line in (prep.SEMCEB_REPO_DIR / "benchmark_queries/queries.jsonl").read_text().splitlines()
            if line.strip()
            for q in [json.loads(line)]
        }
        cls.gt = prep.load_ground_truth_index()

    def test_exact_benchmark_coverage_and_reference_preservation(self):
        products, reviews = prep.export_queries(self.gt)
        expected = {
            case_id
            for case_id, q in self.sources.items()
            if "filter" in q and len(q["datasets"]) == 1
            and prep.IMAGE_COL not in [c.split(".")[-1] for c in prep.PLACEHOLDER_RE.findall(q["filter"])]
        }
        self.assertEqual((len(products), len(reviews)), (30, 19))
        self.assertEqual(set(self.mappings), expected)
        for q in products + reviews:
            with self.subTest(case_id=q["case_id"]):
                source = self.sources[q["case_id"]]
                self.assertEqual(q["source_filter"], source["filter"])
                self.assertEqual(q["raw_filter"], prep.strip_alias_placeholders(source["filter"]))
                self.assertEqual(q["category"], source.get("category", []))
                self.assertEqual(q["type"], source["type"])
                self.assertEqual(q["predicates"], [q["query"]])
                ds = prep.re.split(r"\s+as\s+", source["datasets"][0], flags=prep.re.IGNORECASE)[0].strip()
                gt = self.gt.get((ds, q["raw_filter"])) or self.gt[(ds, source["filter"])]
                self.assertEqual(q["ground_truth_cardinality"], gt["cardinality"])
                self.assertEqual(q["ground_truth_selectivity"], gt["selectivity"])
                stated_fields = prep.re.findall(r"\(([a-z_]+)\)", q["query"])
                self.assertEqual(stated_fields, list(dict.fromkeys(q["columns"])))

    def test_changed_source_requires_review(self):
        q = self.sources["Q009"]
        with self.assertRaisesRegex(ValueError, "Source filter changed"):
            prep.scoped_predicate("Q009", q["filter"].replace("one pound", "two pounds"),
                                  ["features_json"], self.mappings)

    def test_wrong_scope_or_missing_mapping_is_rejected(self):
        q = self.sources["Q009"]
        with self.assertRaisesRegex(ValueError, "Field scope mismatch"):
            prep.scoped_predicate("Q009", q["filter"], ["details_json"], self.mappings)
        with self.assertRaisesRegex(ValueError, "Missing field-scoped predicate"):
            prep.scoped_predicate("Q009", q["filter"], ["features_json"], {})
        mappings = copy.deepcopy(self.mappings)
        mappings["Q009"]["predicate"] = "The package weight is less than one pound."
        with self.assertRaisesRegex(ValueError, "does not state every source field"):
            prep.scoped_predicate("Q009", q["filter"], ["features_json"], mappings)

    def test_duplicate_and_unused_mappings_are_rejected(self):
        payload = json.loads(prep.PREDICATES_PATH.read_text())
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "predicates.json"
            payload["queries"].append(copy.deepcopy(payload["queries"][0]))
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "Duplicate predicate mapping"):
                prep.load_scoped_predicates(path)
            payload["queries"][-1]["case_id"] = "Q999"
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "Unused predicate mappings"):
                prep.export_queries(self.gt, path)


if __name__ == "__main__":
    unittest.main()
