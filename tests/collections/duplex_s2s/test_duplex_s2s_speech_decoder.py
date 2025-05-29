import pytest
import torch
from lhotse import CutSet, SupervisionSegment
from lhotse.testing.dummies import dummy_cut, dummy_recording

from nemo.collections.duplex_s2s.data import DuplexS2SDataset
from nemo.collections.duplex_s2s.models import DuplexS2SSpeechDecoderModel
from nemo.collections.duplex_s2s.parts.testing import as_bfloat16


@pytest.fixture(scope="session")
def model():
    cfg = {
        "pretrained_asr": "stt_en_fastconformer_hybrid_large_streaming_80ms",
        "scoring_asr": "stt_en_fastconformer_transducer_large",
        "pretrained_llm": "TinyLlama/TinyLlama_v1.1",
        "pretrained_audio_codec": "nvidia/low-frame-rate-speech-codec-22khz",
        "pretrained_weights": False,
        "freeze_params": ["^audio_codec\\..+$"],
        "audio_loss_weight": 1,
        "text_loss_weight": 3,
        "perception": {
            "_target_": "nemo.collections.duplex_s2s.modules.perception.AudioPerceptionModule",
            "modality_adapter": {
                "_target_": "nemo.collections.asr.modules.ConformerEncoder",
                "feat_in": 512,
                "feat_out": -1,
                "n_layers": 1,
                "d_model": 512,
                "subsampling_factor": 1,
            },
        },
        "speech_decoder": {
            "n_layers": 1,
            "d_model": 768,
            "d_ffn": 3072,
            "sa_n_heads": 12,
            "kernel_size": 3,
            "is_causal": True,
        },
        "optimizer": {"_target_": "torch.optim.AdamW"},
    }
    return DuplexS2SSpeechDecoderModel(cfg).bfloat16()


@pytest.fixture(scope="session")
def dataset(model):
    return DuplexS2SDataset(
        model.tokenizer,
        frame_length=0.08,
        source_sample_rate=16000,
        target_sample_rate=22050,
        input_roles=["user"],
        output_roles=["assistant"],
    )


@pytest.fixture(scope="session")
def training_cutset_batch():
    cut = dummy_cut(0, recording=dummy_recording(0, with_data=True))
    cut.target_audio = dummy_recording(1, with_data=True)
    cut.supervisions = [
        SupervisionSegment(
            id=cut.id,
            recording_id=cut.recording_id,
            start=0,
            duration=0.1,
            text='hi',
            speaker="user",
        ),
        SupervisionSegment(
            id=cut.id,
            recording_id=cut.recording_id,
            start=0.3,
            duration=0.1,
            text='hello',
            speaker="assistant",
        ),
        SupervisionSegment(
            id=cut.id,
            recording_id=cut.recording_id,
            start=0.5,
            duration=0.1,
            text='ok',
            speaker="user",
        ),
        SupervisionSegment(
            id=cut.id,
            recording_id=cut.recording_id,
            start=0.6,
            duration=0.4,
            text='okay',
            speaker="assistant",
        ),
    ]
    return CutSet([cut])


def test_s2s_speech_decoder_training_step(model, dataset, training_cutset_batch):
    batch = as_bfloat16(dataset[training_cutset_batch])
    results = model.training_step(batch, batch_idx=0)
    assert torch.is_tensor(results["loss"])
    assert not torch.isnan(results["loss"])
    assert results["loss"] > 0


def test_s2s_speech_decoder_offline_generation(model):
    # 16000 samples == 1 second == 12.5 frames ~= 14 frames after encoder padding
    gen_text, gen_audio, lengths = model.offline_inference(
        input_signal=torch.randn(1, 16000),
        input_signal_lens=torch.tensor([16000]),
    )

    assert gen_text.shape == (1, 14)
    assert gen_text.dtype == torch.long
    assert (gen_text >= 0).all()
    assert (gen_text < model.text_vocab_size).all()

    assert gen_audio.shape == (1, 14, 8)
    assert gen_audio.dtype == torch.long
    assert (gen_audio >= 0).all()
    assert (gen_audio < model.speech_vocab_size).all()
