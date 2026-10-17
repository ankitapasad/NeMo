import ast
import logging
from pathlib import Path
from types import SimpleNamespace


def enable(model, enabled=True):
    path = Path(__file__).resolve().parents[3] / 'nemo/collections/speechlm2/models/streaming_stt_model.py'
    cls = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef) and n.name == 'StreamingSTTModel')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_configure_activation_checkpointing')
    namespace = {'logging': logging}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), namespace)
    namespace[method.name](SimpleNamespace(llm=model, core_cfg=SimpleNamespace(llm_gradient_checkpointing=enabled)))


def test_default_leaves_model_unchanged():
    enable(object(), enabled=False)

"""Check LoRA/dropout forward and every trainable gradient with recomputation."""
import copy,torch
from transformers import Qwen3Config,Qwen3ForCausalLM
from peft import LoraConfig,get_peft_model

def test_loss_and_gradient_parity():
    torch.manual_seed(42)
    cfg=Qwen3Config(vocab_size=64,hidden_size=32,intermediate_size=64,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,head_dim=8,max_position_embeddings=64,use_cache=False,attention_dropout=0.1)
    base=get_peft_model(Qwen3ForCausalLM(cfg),LoraConfig(task_type='CAUSAL_LM',r=4,lora_alpha=8,lora_dropout=.01,target_modules=['q_proj','v_proj']))
    for n,p in base.named_parameters():
        if 'embed_tokens' in n or 'lm_head' in n:p.requires_grad_(True)
        if 'lora_B' in n:torch.nn.init.normal_(p,std=.01)
    recompute=copy.deepcopy(base);enable(recompute); assert recompute.is_gradient_checkpointing; assert [(n,p.requires_grad) for n,p in base.named_parameters()]==[(n,p.requires_grad) for n,p in recompute.named_parameters()]
    inputs=torch.randint(0,64,(2,23));out=[]
    for model in [base,recompute]:
        model.train();torch.manual_seed(101);loss=model(input_ids=inputs,labels=inputs,use_cache=False).loss;loss.backward();out.append((loss.detach(),{n:p.grad.detach().clone() for n,p in model.named_parameters() if p.grad is not None}))
    torch.testing.assert_close(out[0][0],out[1][0],rtol=1e-5,atol=1e-6)
    assert out[0][1].keys()==out[1][1].keys()
    for n,g in out[0][1].items():torch.testing.assert_close(g,out[1][1][n],rtol=1e-4,atol=1e-6,msg=n)
