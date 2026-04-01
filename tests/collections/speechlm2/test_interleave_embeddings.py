# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Tests for interleave_embeddings — the pure-tensor function that merges
text and audio embeddings guided by an audio mask.

All tests use small synthetic tensors with distinguishable values so
placements can be verified exactly.
"""

import pytest
import torch

from nemo.collections.speechlm2.data.streaming_stt_dataset import AUDIO_TOKEN_IDX
from nemo.collections.speechlm2.models.streaming_stt_model import interleave_embeddings

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PAD_ID = 0
H = 4  # hidden dim used throughout tests

AUD = AUDIO_TOKEN_IDX  # shorthand (-100)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _constant_embed(value: float, length: int) -> torch.Tensor:
    """Return a (1, length, H) tensor filled with *value*."""
    return torch.full((1, length, H), value)


def _run(input_tokens, text_embeds, audio_embs):
    """Convenience wrapper: adds audio_mask and calls interleave_embeddings."""
    audio_mask = input_tokens == AUD
    return interleave_embeddings(
        input_tokens=input_tokens,
        audio_mask=audio_mask,
        text_embeds=text_embeds,
        audio_embs=audio_embs,
        pad_id=PAD_ID,
    )


# ===========================================================================
# Tests
# ===========================================================================


class TestOutputShape:

    def test_shape_matches_input(self):
        """Output embeds have the same (B, L, H) shape as text_embeds."""
        B, L = 2, 6
        input_tokens = torch.tensor(
            [
                [PAD_ID, AUD, 5, AUD, 7, 8],
                [AUD, AUD, 3, 4, AUD, 6],
            ]
        )
        text_embeds = torch.randn(B, L, H)
        audio_embs = torch.randn(B, 3, H)  # up to 3 audio frames per sample
        result = _run(input_tokens, text_embeds, audio_embs)

        assert result["input_embeds"].shape == (B, L, H)
        assert result["attention_mask"].shape == (B, L)


class TestTextPositions:

    def test_text_embeddings_preserved(self):
        """Non-audio, non-padding positions keep their text embeddings exactly."""
        # tokens: [PAD, text=5, AUD, text=7]
        input_tokens = torch.tensor([[PAD_ID, 5, AUD, 7]])
        text_embeds = torch.tensor([[[0.0] * H, [1.0] * H, [9.9] * H, [3.0] * H]])
        audio_embs = torch.tensor([[[2.0] * H]])  # 1 audio frame

        result = _run(input_tokens, text_embeds, audio_embs)
        embeds = result["input_embeds"]

        # Text positions (indices 1 and 3) should be unchanged.
        assert torch.equal(embeds[0, 1], text_embeds[0, 1])
        assert torch.equal(embeds[0, 3], text_embeds[0, 3])

    def test_text_at_audio_position_overwritten(self):
        """Whatever text_embeds had at audio positions is replaced."""
        input_tokens = torch.tensor([[AUD, 5]])
        text_embeds = torch.tensor([[[9.9] * H, [1.0] * H]])
        audio_embs = torch.tensor([[[2.0] * H]])

        result = _run(input_tokens, text_embeds, audio_embs)
        # Position 0 is audio — should NOT be 9.9
        assert torch.equal(result["input_embeds"][0, 0], torch.tensor([2.0] * H))


class TestAudioPlacement:

    def test_sequential_frame_mapping(self):
        """Audio positions map to frames 0, 1, 2, … in order."""
        # tokens: [AUD, text, AUD, AUD]  → frames 0, -, 1, 2
        input_tokens = torch.tensor([[AUD, 5, AUD, AUD]])
        text_embeds = torch.zeros(1, 4, H)
        # 3 distinct audio frames
        audio_embs = torch.tensor([[[1.0] * H, [2.0] * H, [3.0] * H]])

        result = _run(input_tokens, text_embeds, audio_embs)
        embeds = result["input_embeds"]

        assert torch.equal(embeds[0, 0], torch.tensor([1.0] * H))  # frame 0
        assert torch.equal(embeds[0, 2], torch.tensor([2.0] * H))  # frame 1
        assert torch.equal(embeds[0, 3], torch.tensor([3.0] * H))  # frame 2

    def test_frame_order_independent_of_position(self):
        """The k-th audio token always gets frame k, regardless of where it sits."""
        # Two samples with audio tokens at different positions
        input_tokens = torch.tensor(
            [
                [AUD, AUD, 5, 6],  # audio at positions 0, 1
                [5, 6, AUD, AUD],  # audio at positions 2, 3
            ]
        )
        text_embeds = torch.zeros(2, 4, H)
        audio_embs = torch.stack(
            [
                torch.tensor([[10.0] * H, [20.0] * H]),
                torch.tensor([[30.0] * H, [40.0] * H]),
            ]
        )  # (2, 2, H)

        result = _run(input_tokens, text_embeds, audio_embs)
        embeds = result["input_embeds"]

        # Sample 0: positions 0→frame0, 1→frame1
        assert torch.equal(embeds[0, 0], torch.tensor([10.0] * H))
        assert torch.equal(embeds[0, 1], torch.tensor([20.0] * H))
        # Sample 1: positions 2→frame0, 3→frame1
        assert torch.equal(embeds[1, 2], torch.tensor([30.0] * H))
        assert torch.equal(embeds[1, 3], torch.tensor([40.0] * H))

    def test_per_sample_frame_index_resets(self):
        """Frame indexing starts at 0 independently for each sample in the batch."""
        input_tokens = torch.tensor(
            [
                [AUD, 5],
                [5, AUD],
            ]
        )
        text_embeds = torch.zeros(2, 2, H)
        audio_embs = torch.tensor(
            [
                [[1.0] * H],  # sample 0: 1 frame
                [[2.0] * H],  # sample 1: 1 frame
            ]
        )  # (2, 1, H)

        result = _run(input_tokens, text_embeds, audio_embs)
        embeds = result["input_embeds"]

        # Both samples use frame 0 for their single audio token
        assert torch.equal(embeds[0, 0], torch.tensor([1.0] * H))
        assert torch.equal(embeds[1, 1], torch.tensor([2.0] * H))


class TestAudioPadding:

    def test_more_audio_tokens_than_encoder_frames(self):
        """When the last chunk exceeds encoder output, extra positions get zeros."""
        # 3 audio tokens but only 2 encoder frames
        input_tokens = torch.tensor([[AUD, AUD, AUD]])
        text_embeds = torch.zeros(1, 3, H)
        audio_embs = torch.tensor([[[1.0] * H, [2.0] * H]])  # only 2 frames

        result = _run(input_tokens, text_embeds, audio_embs)
        embeds = result["input_embeds"]

        assert torch.equal(embeds[0, 0], torch.tensor([1.0] * H))  # frame 0
        assert torch.equal(embeds[0, 1], torch.tensor([2.0] * H))  # frame 1
        assert torch.equal(embeds[0, 2], torch.tensor([0.0] * H))  # padded

    def test_exact_frame_count(self):
        """No padding needed when audio tokens == encoder frames."""
        input_tokens = torch.tensor([[AUD, AUD]])
        text_embeds = torch.zeros(1, 2, H)
        audio_embs = torch.tensor([[[1.0] * H, [2.0] * H]])

        result = _run(input_tokens, text_embeds, audio_embs)
        embeds = result["input_embeds"]

        assert torch.equal(embeds[0, 0], torch.tensor([1.0] * H))
        assert torch.equal(embeds[0, 1], torch.tensor([2.0] * H))

    def test_more_encoder_frames_than_audio_tokens(self):
        """Extra encoder frames are simply unused (not an error)."""
        input_tokens = torch.tensor([[AUD, 5]])
        text_embeds = torch.zeros(1, 2, H)
        # 5 encoder frames but only 1 audio token
        audio_embs = torch.randn(1, 5, H)

        result = _run(input_tokens, text_embeds, audio_embs)
        embeds = result["input_embeds"]

        # Only frame 0 is used
        assert torch.equal(embeds[0, 0], audio_embs[0, 0])


class TestAttentionMask:

    def test_padding_masked_out(self):
        """Left-padding positions (pad_id) get attention_mask=False."""
        input_tokens = torch.tensor([[PAD_ID, PAD_ID, 5, AUD, 7]])
        text_embeds = torch.zeros(1, 5, H)
        audio_embs = torch.zeros(1, 1, H)

        result = _run(input_tokens, text_embeds, audio_embs)
        mask = result["attention_mask"]

        assert mask.tolist() == [[False, False, True, True, True]]

    def test_audio_positions_attended(self):
        """Audio positions must have attention_mask=True."""
        input_tokens = torch.tensor([[AUD, AUD, 5]])
        text_embeds = torch.zeros(1, 3, H)
        audio_embs = torch.zeros(1, 2, H)

        result = _run(input_tokens, text_embeds, audio_embs)
        mask = result["attention_mask"]

        assert mask[0, 0].item() is True
        assert mask[0, 1].item() is True

    def test_all_positions_attended_no_padding(self):
        """When there is no padding, all positions are attended."""
        input_tokens = torch.tensor([[5, AUD, 7, AUD]])
        text_embeds = torch.zeros(1, 4, H)
        audio_embs = torch.zeros(1, 2, H)

        result = _run(input_tokens, text_embeds, audio_embs)
        assert result["attention_mask"].all()


class TestAudioOnly:

    def test_all_audio_no_padding(self):
        """All positions are audio — every position gets the corresponding frame."""
        input_tokens = torch.tensor([[AUD, AUD, AUD]])
        text_embeds = torch.full((1, 3, H), 99.0)  # should be entirely overwritten
        audio_embs = torch.tensor([[[1.0] * H, [2.0] * H, [3.0] * H]])

        result = _run(input_tokens, text_embeds, audio_embs)
        embeds = result["input_embeds"]

        assert torch.equal(embeds[0, 0], torch.tensor([1.0] * H))
        assert torch.equal(embeds[0, 1], torch.tensor([2.0] * H))
        assert torch.equal(embeds[0, 2], torch.tensor([3.0] * H))
        assert result["attention_mask"].all()

    def test_all_audio_with_left_padding(self):
        """Left-padding + all-audio content positions."""
        input_tokens = torch.tensor([[PAD_ID, PAD_ID, AUD, AUD]])
        text_embeds = torch.zeros(1, 4, H)
        audio_embs = torch.tensor([[[5.0] * H, [6.0] * H]])

        result = _run(input_tokens, text_embeds, audio_embs)
        embeds = result["input_embeds"]
        mask = result["attention_mask"]

        assert torch.equal(embeds[0, 2], torch.tensor([5.0] * H))
        assert torch.equal(embeds[0, 3], torch.tensor([6.0] * H))
        assert mask.tolist() == [[False, False, True, True]]


class TestNoAudioTokens:

    def test_text_only_passthrough(self):
        """With no audio tokens, output equals text_embeds exactly."""
        input_tokens = torch.tensor([[5, 6, 7]])
        text_embeds = torch.randn(1, 3, H)
        audio_embs = torch.empty(1, 0, H)  # no frames

        result = _run(input_tokens, text_embeds, audio_embs)
        assert torch.equal(result["input_embeds"], text_embeds)


class TestBatched:

    def test_different_audio_counts_per_sample(self):
        """Samples with different numbers of audio tokens are handled correctly."""
        input_tokens = torch.tensor(
            [
                [PAD_ID, AUD, 5, 6],  # 1 audio token (left-padded)
                [AUD, AUD, AUD, 5],  # 3 audio tokens
            ]
        )
        text_embeds = torch.zeros(2, 4, H)
        text_embeds[0, 2] = torch.tensor([10.0] * H)  # text at position 2, sample 0
        text_embeds[1, 3] = torch.tensor([20.0] * H)  # text at position 3, sample 1

        audio_embs = torch.tensor(
            [
                [[1.0] * H, [0.0] * H, [0.0] * H],  # sample 0: 1 real frame
                [[3.0] * H, [4.0] * H, [5.0] * H],  # sample 1: 3 frames
            ]
        )

        result = _run(input_tokens, text_embeds, audio_embs)
        embeds = result["input_embeds"]
        mask = result["attention_mask"]

        # Sample 0: pad, audio(frame0), text=10, text=0
        assert torch.equal(embeds[0, 1], torch.tensor([1.0] * H))
        assert torch.equal(embeds[0, 2], torch.tensor([10.0] * H))
        assert mask[0].tolist() == [False, True, True, True]

        # Sample 1: audio(frame0), audio(frame1), audio(frame2), text=20
        assert torch.equal(embeds[1, 0], torch.tensor([3.0] * H))
        assert torch.equal(embeds[1, 1], torch.tensor([4.0] * H))
        assert torch.equal(embeds[1, 2], torch.tensor([5.0] * H))
        assert torch.equal(embeds[1, 3], torch.tensor([20.0] * H))
        assert mask[1].tolist() == [True, True, True, True]


class TestRightPadding:
    """Verify that right-padding (pad on the right side) is handled correctly."""

    def test_right_padding_masked_out(self):
        """Right-padding positions get attention_mask=False."""
        input_tokens = torch.tensor([[5, AUD, AUD, 7, PAD_ID, PAD_ID]])
        text_embeds = torch.zeros(1, 6, H)
        audio_embs = torch.tensor([[[1.0] * H, [2.0] * H]])

        result = _run(input_tokens, text_embeds, audio_embs)
        mask = result["attention_mask"]
        assert mask.tolist() == [[True, True, True, True, False, False]]

    def test_right_padding_audio_correct(self):
        """Audio frames map correctly with right-padding."""
        input_tokens = torch.tensor([[AUD, AUD, AUD, 5, PAD_ID, PAD_ID]])
        text_embeds = torch.zeros(1, 6, H)
        audio_embs = torch.tensor([[[1.0] * H, [2.0] * H, [3.0] * H]])

        result = _run(input_tokens, text_embeds, audio_embs)
        embeds = result["input_embeds"]
        assert torch.equal(embeds[0, 0], torch.tensor([1.0] * H))
        assert torch.equal(embeds[0, 1], torch.tensor([2.0] * H))
        assert torch.equal(embeds[0, 2], torch.tensor([3.0] * H))

    def test_right_padding_batch_mixed(self):
        """Batch with right-padded shorter sample and full-length sample."""
        input_tokens = torch.tensor(
            [
                [10, AUD, AUD, 20, PAD_ID, PAD_ID],
                [30, AUD, AUD, AUD, AUD, 40],
            ]
        )
        text_embeds = torch.zeros(2, 6, H)
        audio_embs = torch.tensor(
            [
                [[1.0] * H, [2.0] * H, [0.0] * H, [0.0] * H],
                [[3.0] * H, [4.0] * H, [5.0] * H, [6.0] * H],
            ]
        )

        result = _run(input_tokens, text_embeds, audio_embs)
        embeds = result["input_embeds"]
        mask = result["attention_mask"]

        # Sample 0: audio at positions 1,2
        assert torch.equal(embeds[0, 1], torch.tensor([1.0] * H))
        assert torch.equal(embeds[0, 2], torch.tensor([2.0] * H))
        assert mask[0].tolist() == [True, True, True, True, False, False]

        # Sample 1: audio at positions 1,2,3,4
        assert torch.equal(embeds[1, 1], torch.tensor([3.0] * H))
        assert torch.equal(embeds[1, 4], torch.tensor([6.0] * H))
        assert mask[1].tolist() == [True, True, True, True, True, True]


class TestDifferentAudioLengths:
    """Verify correct behavior when samples have different numbers of audio tokens,
    as happens with chunk_size=-1 (offline mode) where num_frames varies per sample."""

    def test_no_audio_leakage_into_padding(self):
        """Padding positions must never contain audio embeddings."""
        input_tokens = torch.tensor(
            [
                [PAD_ID, PAD_ID, PAD_ID, AUD, AUD, 10],
                [AUD, AUD, AUD, AUD, AUD, 20],
            ]
        )
        text_embeds = torch.full((2, 6, H), -1.0)  # fill with -1 so leakage is visible
        audio_embs = torch.full((2, 5, H), 99.0)  # fill with 99

        result = _run(input_tokens, text_embeds, audio_embs)
        embeds = result["input_embeds"]

        # Sample 0 padding positions should have text_embeds (-1), not audio (99)
        for pos in [0, 1, 2]:
            assert (
                embeds[0, pos, 0].item() == -1.0
            ), f"Padding position {pos} leaked audio: got {embeds[0, pos, 0].item()}"

    def test_right_padded_different_audio_counts(self):
        """Right-padded batch where samples have different audio token counts."""
        # Sample 0 (shorter): [text, AUD, AUD, text, PAD, PAD]
        # Sample 1 (longer):  [text, AUD, AUD, AUD, AUD, text]
        input_tokens = torch.tensor(
            [
                [10, AUD, AUD, 20, PAD_ID, PAD_ID],
                [30, AUD, AUD, AUD, AUD, 40],
            ]
        )
        text_embeds = torch.zeros(2, 6, H)
        audio_embs = torch.tensor(
            [
                [[1.0] * H, [2.0] * H, [0.0] * H, [0.0] * H],
                [[3.0] * H, [4.0] * H, [5.0] * H, [6.0] * H],
            ]
        )

        result = _run(input_tokens, text_embeds, audio_embs)
        embeds = result["input_embeds"]
        mask = result["attention_mask"]

        # Sample 0
        assert torch.equal(embeds[0, 1], torch.tensor([1.0] * H))
        assert torch.equal(embeds[0, 2], torch.tensor([2.0] * H))
        assert mask[0].tolist() == [True, True, True, True, False, False]

        # Sample 1
        assert torch.equal(embeds[1, 1], torch.tensor([3.0] * H))
        assert torch.equal(embeds[1, 4], torch.tensor([6.0] * H))
        assert mask[1].tolist() == [True, True, True, True, True, True]


class TestDocstringExample:
    """
    Mirrors the dataset docstring example:
    7 chunks × 2 audio tokens = 14 audio positions, interleaved with
    assistant turn tokens and formatting tokens.
    """

    def test_audio_frame_count(self):
        """All 14 audio positions get distinct frame embeddings."""
        N_AUDIO = 14
        # Simplified token sequence: [header, aud, aud, asst_tok, header, aud, aud, asst_tok, ...]
        # We only care about audio vs non-audio, so use a simple layout:
        # 7 chunks, each chunk = [AUD, AUD, text_tok]
        chunks = []
        for i in range(7):
            chunks.extend([AUD, AUD, 50 + i])  # 50+i = assistant/formatting token
        input_tokens = torch.tensor([chunks])  # (1, 21)
        L = input_tokens.shape[1]

        text_embeds = torch.zeros(1, L, H)
        # 14 unique audio frames
        audio_embs = torch.arange(N_AUDIO).unsqueeze(0).unsqueeze(-1).expand(1, N_AUDIO, H).float()

        result = _run(input_tokens, text_embeds, audio_embs)
        embeds = result["input_embeds"]

        # Verify each audio position got the right frame
        audio_positions = (input_tokens[0] == AUD).nonzero(as_tuple=True)[0]
        assert len(audio_positions) == N_AUDIO
        for frame_idx, pos in enumerate(audio_positions):
            expected = float(frame_idx)
            assert embeds[0, pos, 0].item() == expected, (
                f"Position {pos}: expected frame {frame_idx} (val={expected}), " f"got {embeds[0, pos, 0].item()}"
            )
