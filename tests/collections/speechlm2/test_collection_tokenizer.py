"""CPU tokenizer registration preserves training token IDs and resize order."""
import ast
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest


MODEL = Path(__file__).resolve().parents[3]/'nemo/collections/speechlm2/models/streaming_stt_model.py'


class Tokenizer:
    def __init__(self, *args, **kwargs):
        self.vocab = {'<|im_start|>': 0}
        self.tokenizer = self
    def __len__(self):
        return len(self.vocab)
    def add_special_tokens(self, mapping):
        for token in mapping['additional_special_tokens']:
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)
    def text_to_ids(self, token):
        return [self.vocab[token]]


def setup_function_from_source():
    # No model, remote tokenizer, CUDA or heavyweight parent imports in this unit test.
    node = next(n for n in ast.parse(MODEL.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == 'setup_streaming_tokenizer')
    namespace = dict(AutoTokenizer=Tokenizer, logging=logging,
                     token_in_vocab=lambda token, tokenizer: token in tokenizer.vocab,
                     USER_BACKCHANNEL_MODE_SOB_EOU='sob_eou', USER_BACKCHANNEL_MODE_SOB_EOB='sob_eob')
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(MODEL), 'exec'), namespace)
    return namespace['setup_streaming_tokenizer']


@pytest.mark.parametrize('mode', ['ignore', 'sob_eou', 'sob_eob'])
@pytest.mark.parametrize('no_blank', [False, True])
@pytest.mark.parametrize('agent', [False, True])
def test_cpu_and_training_tokenizer_registration(mode, no_blank, agent):
    setup = setup_function_from_source()
    cfg = SimpleNamespace(pretrained_llm='unused', blank_token='<blank>',
        compact_text_end_only_no_blank=no_blank, use_text_tokens=True, compact_template=True,
        text_start_token='<|text_start|>', text_end_token='<|text_end|>', write_token='<|im_start|>',
        add_utterance_boundary_tokens=True, utterance_start_token='<sou>', utterance_end_token='<eou>',
        user_backchannel_mode=mode, user_backchannel_start_token='<sob>', user_backchannel_end_token='<eob>',
        enable_agent_backchannels=agent, agent_backchannel_start_token='<soab>', agent_backchannel_end_token='<eoab>')
    cpu, blank = setup(cfg)
    resizes = []
    train, train_blank = setup(cfg, resize_embeddings=resizes.append)
    groups = ([] if no_blank else [['<blank>']]) + [['<sou>', '<eou>']]
    if mode != 'ignore':
        groups += [['<sob>'] if mode == 'sob_eou' else ['<sob>', '<eob>']]
    if agent:
        groups += [['<soab>', '<eoab>']]
    groups += [['<|text_end|>'] if no_blank else ['<|text_start|>', '<|text_end|>']]
    expected = ['<|im_start|>']
    expected_resizes = []
    for group in groups:
        expected.extend(group); expected_resizes.append(len(expected))
    assert list(cpu.vocab) == expected
    assert cpu.vocab == train.vocab
    assert resizes == expected_resizes  # Preserve repeated training resize/initialization order.
    assert blank == train_blank == ('' if no_blank else '<blank>')
