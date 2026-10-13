import unittest
from urllib.parse import urlsplit

from app.robustness.mutations import MUTATION_TYPES, generate_mutations


class MutationEngineTests(unittest.TestCase):
    def test_all_categories_generate_unique_synthetic_urls(self):
        cases = generate_mutations(
            "https://example.com:8443/path?next=a%2Fb",
            list(MUTATION_TYPES),
            limit=30,
        )
        urls = [case["url"] for case in cases]
        self.assertEqual(len(urls), len(set(urls)))
        self.assertTrue(all(url.startswith("https://") and urlsplit(url).port == 8443 for url in urls))
        self.assertEqual({case["mutation_type"] for case in cases}, set(MUTATION_TYPES))

    def test_rejects_malformed_or_unsupported_urls(self):
        malformed = ["", "https:///path", "ftp://example.com", "https://user:pass@example.com", "https://example.com:bad/", "https://bad host/path"]
        for url in malformed:
            with self.subTest(url=url), self.assertRaises(ValueError):
                generate_mutations(url, list(MUTATION_TYPES))

    def test_unicode_encoded_and_subdomain_urls_are_supported(self):
        cases = generate_mutations("https://münich.example.com/%E2%9C%93?x=%C3%A9", list(MUTATION_TYPES))
        self.assertTrue(cases)
        self.assertTrue(any("%E2%9C%93" in case["url"] for case in cases))
        self.assertTrue(all(urlsplit(case["url"]).hostname for case in cases))
        self.assertTrue(any(case["mutation_type"] == "subdomain_manipulation" for case in cases))

    def test_limit_and_token_mutations(self):
        cases = generate_mutations("https://example.com", list(MUTATION_TYPES), limit=4)
        self.assertEqual(len(cases), 4)
        token_cases = generate_mutations("https://example.com", ["token_manipulation"])
        self.assertEqual(len(token_cases), 5)
        self.assertTrue(all(case["mutation_type"] == "token_manipulation" for case in token_cases))


class RobustnessEndpointTests(unittest.TestCase):
    def test_reuses_offline_prediction_and_reports_metrics(self):
        from app.api import main

        calls = []

        def fake_predict(url):
            calls.append(url)
            phishing = "login" in url
            return {"prediction": "phishing" if phishing else "legitimate", "confidence": .91, "phishing_probability": .91 if phishing else .09}

        original_predict = main.offline_predict
        main.offline_predict = fake_predict
        try:
            result = main.robustness_test(main.RobustnessRequest(
                url="https://example.com", num_mutations=3,
                mutation_types=["token_manipulation"],
            ))
        finally:
            main.offline_predict = original_predict

        self.assertEqual(len(calls), 4)
        self.assertEqual(result["metrics"]["total_variants"], 3)
        self.assertGreaterEqual(result["metrics"]["prediction_changes"], 1)
        self.assertEqual(result["category_metrics"][0]["mutation_type"], "token_manipulation")
        self.assertEqual(result["original"]["confidence"], .91)

    def test_api_rejects_mutation_count_above_limit(self):
        from app.api.main import RobustnessRequest
        from pydantic import ValidationError

        with self.assertRaises(ValidationError):
            RobustnessRequest(url="https://example.com", num_mutations=31)


if __name__ == "__main__":
    unittest.main()
