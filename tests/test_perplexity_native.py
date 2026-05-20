# ruff: noqa: E402
"""Tests for the dual-backend Perplexity source (native API + OpenRouter fallback)."""

import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "skills" / "last30days" / "scripts"))

from lib import perplexity, pipeline


class BackendSelectionTests(unittest.TestCase):
    def test_prefers_native_perplexity_over_openrouter(self):
        sel = perplexity._select_backend(
            {"PERPLEXITY_API_KEY": "px", "OPENROUTER_API_KEY": "or"}
        )
        self.assertEqual(("perplexity", "px"), sel)

    def test_falls_back_to_openrouter(self):
        sel = perplexity._select_backend({"OPENROUTER_API_KEY": "or"})
        self.assertEqual(("openrouter", "or"), sel)

    def test_none_when_no_key(self):
        self.assertIsNone(perplexity._select_backend({}))


class DateFilterTests(unittest.TestCase):
    def test_iso_to_mmddyyyy(self):
        self.assertEqual("04/30/2026", perplexity._to_mmddyyyy("2026-04-30"))


class CitationExtractionTests(unittest.TestCase):
    def test_native_prefers_search_results(self):
        data = {
            "search_results": [
                {"title": "A", "url": "https://a.com", "date": "2026-05-01"},
                {"title": "B", "url": "https://b.com"},
            ],
            "citations": ["https://ignored.com"],
        }
        out = perplexity._extract_citations("perplexity", data, {})
        self.assertEqual(
            [{"url": "https://a.com", "title": "A"}, {"url": "https://b.com", "title": "B"}],
            out,
        )

    def test_native_falls_back_to_citations_list(self):
        data = {"citations": ["https://a.com", "https://a.com", "https://b.com"]}
        out = perplexity._extract_citations("perplexity", data, {})
        self.assertEqual(
            [{"url": "https://a.com", "title": ""}, {"url": "https://b.com", "title": ""}],
            out,
        )

    def test_openrouter_reads_annotations(self):
        choice = {
            "message": {
                "annotations": [
                    {"url_citation": {"url": "https://a.com", "title": "A"}},
                    {"url_citation": {"url": "https://a.com", "title": "dup"}},
                ]
            }
        }
        out = perplexity._extract_citations("openrouter", {}, choice)
        self.assertEqual([{"url": "https://a.com", "title": "A"}], out)


class SearchDispatchTests(unittest.TestCase):
    def _native_response(self):
        return {
            "choices": [{"message": {"content": "Synthesis text."}}],
            "search_results": [{"title": "Src", "url": "https://src.com", "date": "2026-05-02"}],
        }

    def _openrouter_response(self):
        return {
            "choices": [
                {
                    "message": {
                        "content": "Synthesis text.",
                        "annotations": [{"url_citation": {"url": "https://src.com", "title": "Src"}}],
                    }
                }
            ]
        }

    def test_native_path_hits_perplexity_with_date_filters(self):
        with mock.patch.object(perplexity.http, "post", return_value=self._native_response()) as post:
            items, artifact = perplexity.search(
                "openclaw", ("2026-04-01", "2026-04-30"), {"PERPLEXITY_API_KEY": "px"}
            )
        url, payload = post.call_args.args[0], post.call_args.args[1]
        self.assertEqual(perplexity.PERPLEXITY_URL, url)
        self.assertEqual("sonar-pro", payload["model"])
        self.assertEqual("04/01/2026", payload["search_after_date_filter"])
        self.assertEqual("04/30/2026", payload["search_before_date_filter"])
        self.assertEqual("perplexity", artifact["backend"])
        # PX1 synthesis + one citation item
        self.assertEqual("https://src.com", items[1]["url"])

    def test_openrouter_path_hits_openrouter_without_date_filters(self):
        with mock.patch.object(perplexity.http, "post", return_value=self._openrouter_response()) as post:
            items, artifact = perplexity.search(
                "openclaw", ("2026-04-01", "2026-04-30"), {"OPENROUTER_API_KEY": "or"}
            )
        url, payload = post.call_args.args[0], post.call_args.args[1]
        self.assertEqual(perplexity.OPENROUTER_URL, url)
        self.assertEqual("perplexity/sonar-pro", payload["model"])
        self.assertNotIn("search_after_date_filter", payload)
        self.assertEqual("openrouter", artifact["backend"])
        self.assertEqual("https://src.com", items[1]["url"])

    def test_no_key_skips_cleanly(self):
        items, artifact = perplexity.search("openclaw", ("2026-04-01", "2026-04-30"), {})
        self.assertEqual([], items)
        self.assertEqual({}, artifact)


class PipelineAvailabilityTests(unittest.TestCase):
    def test_perplexity_available_with_native_key(self):
        available = pipeline.available_sources(
            {"PERPLEXITY_API_KEY": "px", "INCLUDE_SOURCES": ""},
            requested_sources=["perplexity"],
        )
        self.assertIn("perplexity", available)

    def test_perplexity_still_available_with_openrouter_key(self):
        available = pipeline.available_sources(
            {"OPENROUTER_API_KEY": "or", "INCLUDE_SOURCES": ""},
            requested_sources=["perplexity"],
        )
        self.assertIn("perplexity", available)


if __name__ == "__main__":
    unittest.main()
