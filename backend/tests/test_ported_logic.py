import unittest

from cortex.chunking.chunker import (
    find_sentence_boundary,
)
from cortex.domain_models.facts_ledger import FactsLedger, ParticipantAnalysis
from cortex.llm.runtime import _try_load_json


class TestPortedLogic(unittest.TestCase):
    # -------------------------------------------------------------------------
    # 1. Robust Chunking Regexes
    # -------------------------------------------------------------------------
    def test_unicode_sentence_boundary(self):
        """Verify sentence splitting handles CJK and other punctuation."""
        # Simple English
        text = "Hello World. This is a test."
        pos = find_sentence_boundary(text, 5, window=20)
        # Should find '.' at index 11. End of sentence is 12 (dot + space).
        # Actually pattern matches `.` + ` ` (space).
        # My regex: SENTENCE_END_PATTERN = re.compile(r"[.!?。！？...؟۔]+" r'[)"\'\u2018\u2019»›）】〔〕〉》」』〗〙〞]*' r"(?:\s+|$)")
        # In "Hello World. This", the match is ". " (dot space). Start at 11, end at 13.
        # find_sentence_boundary searches within window.
        # It should return 13.
        self.assertEqual(pos, 13)

        # CJK
        # text_cjk = "你好。世界！" - kept as comment for context
        # "你好" is 2 chars. "。" is 1 char.
        # Match "。" (and implicit end or next char?).
        # Regex matches `。` followed by whitespace OR end of string.
        # "你好。世界" -> "。" is followed by "世". Does it match?
        # My regex: `(?:\s+|$)`. So it REQUIRES whitespace or end of string.
        # Wait. In CJK, we often don't have space.
        # Let's check reference implementation again.
        # Reference: `SENTENCE_END_PATTERN = re.compile(r"[.!?。！？...؟۔]+" r'[...]*' r"(?:\s+|$)")`
        # It DOES require whitespace or end of string.
        # So "你好。世界" might NOT split if there is no space?
        # That would be a bug/limitation of the reference code if true, or I copied it wrong.
        # Let's verify behavior.
        # "你好。世界" -> dot is '。'. Next is '世'. No space. regex `(?:\s+|$)` will fail.
        # So "你好。世界" won't split.
        # BUT "你好。 世界" (with space) will.
        # Is that intended?
        # The reference file `llm_text_chunker.py` had:
        # `SENTENCE_END_PATTERN = re.compile(r"[.!?。！？...؟۔]+" ... r"(?:\s+|$)")`
        # Yes, it seems to require space or EOS.
        # In modern CJK chunking, we usually split on period regardless of space.
        # However, I am porting the *reference* logic. If reference logic requires space, I'll stick to it strictly for now.
        # Wait, let's verify if `find_sentence_boundary` uses it.
        # `chunk_text` calls `find_sentence_boundary`.

        # Test finding end of string
        pos_cjk = find_sentence_boundary("你好。", 0, window=10)
        self.assertEqual(pos_cjk, 3)  # "你好" (2) + "。" (1) = 3.

    # -------------------------------------------------------------------------
    # 2. Robust JSON Parsing
    # -------------------------------------------------------------------------
    def test_json_parsing_nested_braces(self):
        """Verify parser handles nested braces inside strings correctly."""
        # The greedy regex `m = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", candidate)`
        # often fails on `{"key": "value with } brace"} ` or matching braces.
        # Actually greedy regex `.*` usually swallows everything until the *last* brace.
        # But `balanced` parser is safer for multiple objects or chatty strings.

        # Case: JSON inside chatty text with braces in strings
        text = 'Here is the data: {"summary": "This includes a {brace} inside."} ... extra text.'
        parsed = _try_load_json(text)
        self.assertEqual(parsed["summary"], "This includes a {brace} inside.")

        # Case: Fenced code block
        text_fenced = '```json\n{"a": 1}\n```'
        self.assertEqual(_try_load_json(text_fenced), {"a": 1})

    # -------------------------------------------------------------------------
    # 3. Facts Ledger Merging
    # -------------------------------------------------------------------------
    def test_facts_ledger_merge(self):
        """Verify FactsLedger.merge correctly unions data."""
        l1 = FactsLedger(  # noqa: F841 used for side effect test setup or illustrative? Assuming needs keeping or removal.
            asks=[
                {"description": "Ask 1"}
            ],  # Simplification: passing dicts works if pydantic coerces?
            # No, strict mode usually requires objects. `_merge_lists` handles objects.
            # But `FactsLedger(asks=[...])` expects objects.
            # Need to construct full objects.
        )
        # Re-do this properly with objects
        from cortex.domain_models.facts_ledger import Ask

        f1 = FactsLedger(
            asks=[Ask(description="Do X")],
            participants=[
                ParticipantAnalysis(
                    name="Alice", role="client", email="alice@example.com"
                )
            ],
        )
        f2 = FactsLedger(
            asks=[Ask(description="Do Y")],
            participants=[
                ParticipantAnalysis(
                    name="Alice",
                    role="client",
                    email="alice@example.com",
                    tone="friendly",
                ),
                ParticipantAnalysis(name="Bob", role="broker"),
            ],
        )

        merged = f1.merge(f2)

        # Asks should be 2
        self.assertEqual(len(merged.asks), 2)
        descriptions = {a.description for a in merged.asks}
        self.assertEqual(descriptions, {"Do X", "Do Y"})

        # Participants should be 2 (Alice merged, Bob added)
        self.assertEqual(len(merged.participants), 2)

        alice = next(p for p in merged.participants if p.name == "Alice")
        self.assertEqual(alice.tone, "friendly")  # Should be updated from f2
        self.assertEqual(alice.email, "alice@example.com")


if __name__ == "__main__":
    unittest.main()
