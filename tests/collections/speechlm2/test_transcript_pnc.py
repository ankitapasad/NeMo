import ast
import copy
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).parents[3]
spec = importlib.util.spec_from_file_location("pnc", ROOT / "nemo/collections/speechlm2/parts/transcript_pnc.py")
pnc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pnc)
TOKENS = ("<blank>", "<sou>", "<eou>", "<sob>", "<eob>", "<soab>", "<eoab>",
          "<|text_start|>", "<|text_end|>")


def transform(messages):
    return pnc.remove_transcript_pnc(messages, protected_tokens=TOKENS, blank_token="<blank>")


class TestTranscriptPnC(unittest.TestCase):
    def test_question_and_capitalization(self):
        out = transform([dict(role="assistant", content="<sou>Is This RIGHT?<eou>")])
        self.assertEqual(out[0]["content"], "<sou>is this right <eou>")

    def test_unicode_apostrophes_and_hyphens(self):
        out = transform([dict(role="assistant", content="DON’T re-enter—‘Café!’")])
        self.assertEqual(out[0]["content"].strip(), "dont re enter café")

    def test_all_structural_tokens_survive(self):
        for token in TOKENS:
            message = dict(role="assistant", content=token)
            self.assertEqual(transform([message]), [message])

    def test_prompts_audio_and_inputs_not_mutated(self):
        messages = [dict(role="system", content="Transcribe. Keep CAPS?"),
                    dict(role="user", content="<audio><audio>"),
                    dict(role="assistant", content="<sou>HELLO!")]
        saved = copy.deepcopy(messages)
        result = transform(messages)
        self.assertEqual(messages, saved)
        self.assertEqual(result[:2], saved[:2])
        self.assertEqual(len(result), len(messages))

    def test_punctuation_only_chunk_keeps_slot(self):
        self.assertEqual(transform([dict(role="assistant", content=" ?!… ")])[0]["content"], "<blank>")
        self.assertEqual(transform([dict(role="assistant", content="<sou>?!")])[0]["content"], "<sou> ")

    def test_no_number_verbalization(self):
        out = transform([dict(role="assistant", content="3.5, 10:30 & $20")])
        self.assertEqual(out[0]["content"], "3 5 10 30 $20")

    def test_word_boundary_across_chunks(self):
        out = transform([dict(role="assistant", content="HELLO,"),dict(role="assistant", content=" WORLD?")])
        self.assertEqual([r['content'] for r in out], ["hello ", " world "])

    def test_pipeline_after_alignment_and_boundary_before_tokenization_default_off(self):
        source = (ROOT/'nemo/collections/speechlm2/data/streaming_stt_dataset.py').read_text()
        tree = ast.parse(source)
        cfg = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'StreamingSTTDataConfig')
        field = next(n for n in cfg.body if isinstance(n, ast.AnnAssign) and n.target.id == 'transcript_pnc')
        self.assertIs(field.value.value, True)
        start = source.index('    def get_batch_data(')
        segment = source[start:]
        self.assertLess(segment.index('get_llm_messages_for_batch('), segment.index('if not self.cfg.transcript_pnc:'))
        self.assertLess(segment.index('batch_messages ='), segment.index('if not self.cfg.transcript_pnc:'))
        self.assertLess(segment.index('if not self.cfg.transcript_pnc:'), segment.index('_tokenize_compact_with_assistant_mask('))


if __name__ == '__main__':
    unittest.main()
