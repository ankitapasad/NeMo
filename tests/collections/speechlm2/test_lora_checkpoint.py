# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Real PEFT/Qwen merge, NeMo embedding layout, strict load and full-FT tests."""
from pathlib import Path

import pytest
import torch
from torch import nn
from omegaconf import OmegaConf
from peft import LoraConfig, PeftModel, get_peft_model
from peft.tuners.lora import LoraLayer
from transformers import Qwen3Config, Qwen3ForCausalLM

from nemo.collections.speechlm2.parts.lora_checkpoint import full_finetuning_lora_init
from nemo.collections.speechlm2.parts.pretrained import move_embedding
from nemo.collections.speechlm2.parts.stage2_checkpoint import load_asr_stage2_weights


class TinyTarget(nn.Module):
    """Tiny real Qwen using NeMo's external, tied embedding layout."""
    def __init__(self, vocab=32, dtype=torch.float32, frozen_encoder=True):
        super().__init__()
        self.cfg = OmegaConf.create({'freeze_llm_model': False, 'freeze_speech_encoder': frozen_encoder})
        self.llm = Qwen3ForCausalLM(Qwen3Config(vocab_size=vocab, hidden_size=8, intermediate_size=16,
                                             num_hidden_layers=1, num_attention_heads=2,
                                             num_key_value_heads=1, head_dim=4, tie_word_embeddings=True)).to(dtype)
        self.embed_tokens = self.llm.model.embed_tokens
        del self.llm.model.embed_tokens
        self.perception = nn.Linear(8, 8, bias=False).to(dtype)
        self.user_backchannel_mode = 'sob_eob'
        tokens = ['<sou>', '<eou>', '<sob>', '<eob>', '<|text_start|>', '<|text_end|>']
        ids = {t: vocab - 6 + i for i, t in enumerate(tokens)}
        self.tokenizer = type('Tokenizer', (), {'encode': lambda _, t, **kw: [ids[t]]})()
        self._apply_freeze_config()

    def _apply_freeze_config(self):
        self.llm.requires_grad_(True)
        self.embed_tokens.requires_grad_(True)
        self.perception.requires_grad_(not self.cfg.freeze_speech_encoder)


def adapter_source(vocab=32, dtype=torch.float32):
    torch.manual_seed(42)
    model = TinyTarget(vocab, dtype)
    with move_embedding(model):
        model.llm = get_peft_model(model.llm, LoraConfig(task_type='CAUSAL_LM', r=2, lora_alpha=4,
                                                       target_modules=['q_proj', 'v_proj'], lora_dropout=0.0))
    with torch.no_grad():
        for name, param in model.named_parameters():
            if '.lora_' in name:
                param.copy_(torch.randn_like(param) * 0.1)
    return model


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16, torch.float16])
def test_real_peft_merge_precision_and_external_embeddings(dtype):
    source = adapter_source()
    state = {k: v.detach().clone() for k, v in source.state_dict().items()}
    target = TinyTarget(dtype=dtype)
    with full_finetuning_lora_init(target, rank=2, alpha=4):
        target.load_state_dict(state, strict=True)
        assert isinstance(target.llm, PeftModel)
        assert not hasattr(target.llm.base_model.model.model, 'embed_tokens')
        for module in target.llm.modules():
            if isinstance(module, LoraLayer): assert module.get_base_layer().weight.dtype == torch.float32
    assert not isinstance(target.llm, PeftModel)
    assert not hasattr(target.llm.model, 'embed_tokens')
    assert target.llm.lm_head.weight is target.embed_tokens.weight
    assert not any('.lora_' in k or '.base_layer.' in k for k in target.state_dict())
    assert all(p.requires_grad for p in target.llm.parameters())
    assert not target.perception.weight.requires_grad
    for projection in ['q_proj', 'v_proj']:
        stem = 'llm.base_model.model.model.layers.0.self_attn.' + projection + '.'
        expected = (state[stem+'base_layer.weight'].float() +
                    2 * state[stem+'lora_B.default.weight'].float() @ state[stem+'lora_A.default.weight'].float()).to(dtype)
        actual = getattr(target.llm.model.layers[0].self_attn, projection).weight
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    # Non-adapted tensors are loaded too, not reset to the pretrained base.
    torch.testing.assert_close(target.perception.weight, state['perception.weight'].to(dtype))
    if dtype == torch.float32:
        source.eval(); target.eval()
        ids = torch.tensor([[1, 2, 3]])
        with torch.no_grad(), move_embedding(source), move_embedding(target):
            torch.testing.assert_close(source.llm(ids).logits, target.llm(ids).logits, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize('case', ['rank', 'missing', 'extra', 'nan'])
def test_invalid_checkpoint_rejected(case):
    state = adapter_source().state_dict()
    key = next(k for k in state if 'lora_A' in k)
    if case == 'rank': state[key] = state[key][:1]
    if case == 'missing': del state[key]
    if case == 'extra': state[key.replace('.default.', '.other.')] = state[key]
    if case == 'nan': state[key][0, 0] = float('nan')
    target = TinyTarget()
    with pytest.raises((ValueError, RuntimeError)):
        with full_finetuning_lora_init(target, rank=2, alpha=4):
            target.load_state_dict(state, strict=True)


@pytest.mark.parametrize('case', ['frozen', 'adapter_target', 'rank', 'alpha'])
def test_invalid_target_or_options(case):
    model = adapter_source() if case == 'adapter_target' else TinyTarget()
    rank, alpha = 2, 4
    if case == 'frozen': model.cfg.freeze_llm_model = True
    if case == 'rank': rank = 0
    if case == 'alpha': alpha = float('nan')
    with pytest.raises(ValueError):
        with full_finetuning_lora_init(model, rank=rank, alpha=alpha): pass


@pytest.mark.parametrize('frozen_encoder', [True, False])
def test_asr_vocab_merge_full_weight_update_and_resume(tmp_path, frozen_encoder):
    source = adapter_source(vocab=151672)
    path = tmp_path/'asr.ckpt'; torch.save({'state_dict': source.state_dict(), 'hyper_parameters': {'cfg': {'lora': {'r': 2, 'lora_alpha': 4}}}}, path)
    target = TinyTarget(vocab=151676, frozen_encoder=frozen_encoder)
    boundary = target.embed_tokens.weight[151670:151674].detach().clone()
    load_asr_stage2_weights(target, path)
    torch.testing.assert_close(target.embed_tokens.weight[:151670], source.embed_tokens.weight[:151670])
    torch.testing.assert_close(target.embed_tokens.weight[151670:151674], boundary)
    torch.testing.assert_close(target.embed_tokens.weight[151674:], source.embed_tokens.weight[-2:])
    assert target.perception.weight.requires_grad == (not frozen_encoder)
    params = lambda model: [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params(target), lr=1e-3)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1)
    q = target.llm.model.layers[0].self_attn.q_proj
    old = q.weight.detach().clone(); x = torch.randn(2, 8)
    q(x).square().mean().backward(); opt.step(); sched.step()
    assert not torch.equal(q.weight, old)
    checkpoint = tmp_path/'full.ckpt'
    torch.save({'state_dict': target.state_dict(), 'optimizer': opt.state_dict(), 'scheduler': sched.state_dict()}, checkpoint)
    resumed = TinyTarget(vocab=151676, frozen_encoder=frozen_encoder)
    saved = torch.load(checkpoint, weights_only=False)
    # Full-FT resume uses plain weights and no adapter initialization.
    resumed.load_state_dict(saved['state_dict'], strict=True)
    resumed_opt = torch.optim.AdamW(params(resumed), lr=1e-3)
    resumed_sched = torch.optim.lr_scheduler.StepLR(resumed_opt, step_size=1)
    resumed_opt.load_state_dict(saved['optimizer']); resumed_sched.load_state_dict(saved['scheduler'])
    torch.testing.assert_close(resumed.state_dict(), target.state_dict())
    torch.testing.assert_close(resumed_opt.state_dict(), opt.state_dict())
    assert resumed_sched.state_dict() == sched.state_dict()
    resumed.llm.model.layers[0].self_attn.q_proj(x).square().mean().backward(); resumed_opt.step()
    assert not torch.equal(resumed.llm.model.layers[0].self_attn.q_proj.weight, q.weight)


def test_full_ft_recipe():
    from hydra import compose, initialize_config_dir
    root = Path(__file__).resolve().parents[3]
    with initialize_config_dir(config_dir=str(root/'examples/speechlm2/conf'), version_base=None):
        cfg = compose(config_name='streaming_stt_full_ft')
    assert 'lora' not in cfg.model
    assert not cfg.model.freeze_llm_model
    assert cfg.model.freeze_speech_encoder
    assert cfg.get('init_lora_merge') is None


@pytest.mark.parametrize('with_metadata', [True, False])
def test_existing_adapter_training_initialization_unchanged(tmp_path, with_metadata):
    source = adapter_source(vocab=151672)
    checkpoint = {'state_dict': source.state_dict()}
    if with_metadata: checkpoint['hyper_parameters'] = {'cfg': {'lora': {'r': 2, 'lora_alpha': 4}}}
    path = tmp_path/'asr.ckpt'; torch.save(checkpoint, path)
    target = adapter_source(vocab=151676)
    load_asr_stage2_weights(target, path)
    assert isinstance(target.llm, PeftModel)
    for name, value in source.state_dict().items():
        if '.lora_' in name:
            torch.testing.assert_close(target.state_dict()[name], value)


@pytest.mark.parametrize('saved,fallback,expected', [
    ({'r': 2, 'lora_alpha': 4}, None, {'rank': 2, 'alpha': 4}),
    ({'r': 2}, {'alpha': 4}, {'rank': 2, 'alpha': 4}),
    ({}, {'rank': 2, 'alpha': 4}, {'rank': 2, 'alpha': 4}),
])
def test_source_metadata_resolution(saved, fallback, expected):
    from nemo.collections.speechlm2.parts.stage2_checkpoint import _source_lora_merge_config
    assert _source_lora_merge_config({'hyper_parameters': {'cfg': {'lora': saved}}}, fallback) == expected


@pytest.mark.parametrize('saved,fallback,match', [
    ({'r': 2}, None, 'init_lora_merge.alpha'),
    ({'lora_alpha': 4}, None, 'init_lora_merge.rank'),
    ({'r': 2, 'lora_alpha': 4}, {'alpha': 8}, 'conflicts'),
    ({'r': 2, 'lora_alpha': 4, 'use_rslora': True}, None, 'use_rslora'),
    ({'r': 2, 'lora_alpha': 4, 'use_dora': True}, None, 'use_dora'),
    ({'r': 2, 'lora_alpha': 4, 'alpha_pattern': {'q_proj': 8}}, None, 'alpha_pattern'),
    ({'r': 2, 'lora_alpha': 4, 'target_modules': ['k_proj']}, None, 'q_proj/v_proj'),
])
def test_missing_or_unsupported_source_metadata(saved, fallback, match):
    from nemo.collections.speechlm2.parts.stage2_checkpoint import _source_lora_merge_config
    with pytest.raises(ValueError, match=match):
        _source_lora_merge_config({'hyper_parameters': {'cfg': {'lora': saved}}}, fallback)


def test_plain_asr_source_does_not_merge(tmp_path):
    source = TinyTarget(vocab=151672)
    path = tmp_path/'plain_asr.ckpt'; torch.save({'state_dict': source.state_dict()}, path)
    target = TinyTarget(vocab=151676)
    load_asr_stage2_weights(target, path)
    assert not isinstance(target.llm, PeftModel)
    torch.testing.assert_close(target.llm.model.layers[0].self_attn.q_proj.weight,
                               source.llm.model.layers[0].self_attn.q_proj.weight)


def test_missing_metadata_errors_before_mutating_model(tmp_path):
    source = adapter_source(vocab=151672)
    path = tmp_path/'missing_metadata.ckpt'; torch.save({'state_dict': source.state_dict()}, path)
    target = TinyTarget(vocab=151676)
    with pytest.raises(ValueError, match='init_lora_merge.rank'):
        load_asr_stage2_weights(target, path)
    assert not isinstance(target.llm, PeftModel)
    load_asr_stage2_weights(target, path, lora_merge={'rank': 2, 'alpha': 4})
    assert not isinstance(target.llm, PeftModel)
