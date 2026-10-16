"""CPU contract tests; no model downloads or NeMo optional dependencies."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

path = Path(__file__).resolve().parents[3] / 'nemo/collections/speechlm2/parts/token_probability_capture.py'
spec = importlib.util.spec_from_file_location('capture', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def capture():
    return module.TokenProbabilityCapture({k: {'token_id': i, 'token_piece': f'<{k}>'}
                                          for i, k in enumerate(('sou', 'eou', 'sob', 'eob'))}, .08, 2)


def test_full_vocabulary_not_subset_and_no_mutation():
    logits = torch.tensor([[0., 1., 2., 3., 10.]], dtype=torch.float16)
    original = logits.clone()
    probs, logs = module.selected_probabilities(logits, [0, 1, 2, 3])
    assert torch.equal(logits, original)
    assert probs == pytest.approx(torch.softmax(logits.float(), -1)[0, :4].tolist())
    assert logs == pytest.approx(torch.log_softmax(logits.float(), -1)[0, :4].tolist())
    assert sum(probs) < .01


def test_interleaved_streams_record_every_selected_token_including_nonboundary():
    c = capture()
    streams = [[], []]
    for stream, token, frame in [(0, 0, 2), (1, 4, 2), (0, 3, 4), (1, 2, 2)]:
        c.after_sample(torch.tensor([token]), c.before_sample(torch.zeros(1, 5)))
        c.append(token, frame, streams[stream])
        streams[stream].append(token)
    for stream in streams:
        chunks = c.finish_stream(stream, [str(x) for x in stream])
        points = [p for chunk in chunks for p in chunk['predictions']]
        assert [p['selected_token_id'] for p in points] == stream
        assert [p['prediction_index'] for p in points] == [0, 1]
        assert all(sum(p['probabilities'].values()) == pytest.approx(.8) for p in points)
    c.assert_consumed()


@pytest.mark.parametrize('value', [float('nan'), float('inf'), float('-inf')])
def test_nonfinite_rejected(value):
    with pytest.raises(ValueError, match='nonfinite'):
        module.selected_probabilities(torch.tensor([[value, 1., 2.]]), [0, 1])


def test_forced_or_mismatched_token_rejected():
    c = capture()
    with pytest.raises(ValueError, match='not actually sampled'):
        c.append(0, 2, [])
    c.after_sample(torch.tensor([0]), c.before_sample(torch.zeros(1, 5)))
    with pytest.raises(ValueError, match='mismatch'):
        c.append(1, 2, [])


def test_missing_stream_finalization_rejected():
    c = capture()
    c.after_sample(torch.tensor([0]), c.before_sample(torch.zeros(1, 5)))
    with pytest.raises(ValueError, match='orphaned'):
        c.assert_consumed()


def test_boundary_token_mapping():
    pieces = {'<sou>': 0, '<eou>': 1, '<sob>': 2, '<eob>': 3}
    cfg = SimpleNamespace(user_backchannel_mode='sob_eob',
                          utterance_start_token='<sou>', utterance_end_token='<eou>',
                          user_backchannel_start_token='<sob>', user_backchannel_end_token='<eob>')
    tokenizer = SimpleNamespace(tokenizer=SimpleNamespace(encode=lambda piece, **kw: [pieces[piece]]))
    result = module.tracked_tokens(SimpleNamespace(core_cfg=cfg, tokenizer=tokenizer))
    assert list(result) == ['sou', 'eou', 'sob', 'eob']
