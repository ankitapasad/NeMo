import torch
import yaml
from omegaconf import OmegaConf, DictConfig
import numpy as np
import librosa
import time
from transformers import DynamicCache
import re
import os
import sys
import argparse
import torchaudio

# Update the sys path for your environment
# BASE_DIR = "/lustre/fsw/portfolios/llmservice/users/kevinhu/code/s2s_eartts"
# CODE_DIR_NEW = f"{BASE_DIR}/NeMo"
# sys.path.insert(0, CODE_DIR_NEW)

# CODE_DIR="/lustre/fsw/portfolios/llmservice/users/kevinhu/code/s2s_eartts"
# export PYTHONPATH="${CODE_DIR}:${LHOTSE_DIR}:${PYTHONPATH}"

# Set environment variables
cache_dir="/lustre/fsw/portfolios/llmservice/users/kevinhu/hfcache"
os.environ["HF_HOME"] = cache_dir
os.environ["TORCH_HOME"] = cache_dir
os.environ["NEMO_CACHE_DIR"] = cache_dir

import pdb; pdb.set_trace()

from nemo.collections.speechlm2.models.duplex_s2s_external_speech_decoder_model import DuplexS2SExternalSpeechDecoderModel
from nemo.collections.speechlm2.models.duplex_s2s_model import tokens_to_str
from nemo.collections.speechlm2.parts.precision import fp32_precision
from nemo.collections.audio.parts.utils.resampling import resample

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Streaming Parameters ---
SAMPLE_RATE = 16000
FRAME_SIZE_SEC = 0.08  # 80ms per frame
FRAME_SIZE_SAMPLES = int(SAMPLE_RATE * FRAME_SIZE_SEC)  # 1280 samples


class RealtimeStreamingInference:
    """
    Realtime streaming inference that simulates microphone data capture.
    Uses a sliding window buffer and processes audio frame by frame.
    """
    
    def __init__(self, 
                 model_path: str, 
                 llm_checkpoint_path: str,
                 speaker_reference: str,
                 buffer_size_frames: int = 10):
        """
        Initialize the model for realtime streaming inference.
        
        Args:
            model_path (str): Path to eartts's checkpoint (HF format, contains TTS weights)
            llm_checkpoint_path (str): Path to YOUR checkpoint (HF format, contains correct LLM + perception)
            speaker_reference (str): Path to speaker reference audio file
            buffer_size_frames (int): Size of audio buffer in frames (each frame = 80ms)
        """
        print("=" * 70)
        print("INITIALIZING REALTIME STREAMING INFERENCE")
        print("=" * 70)
        print(f"Frame size: {FRAME_SIZE_SEC}s ({FRAME_SIZE_SAMPLES} samples @ {SAMPLE_RATE}Hz)")
        print(f"Buffer size: {buffer_size_frames} frames ({buffer_size_frames * FRAME_SIZE_SEC}s)")
        print("=" * 70)
        
        self.model_path = model_path
        self.llm_checkpoint_path = llm_checkpoint_path
        self.speaker_reference = speaker_reference
        self.buffer_size_frames = buffer_size_frames
        self.buffer_size_samples = buffer_size_frames * FRAME_SIZE_SAMPLES
        
        self.model = None
        self.tokenizer = None
        self.dtype = None
        
        self._initialize_model()
        
        print(f"\n✅ RealtimeStreamingInference initialized successfully.")
        
    def _load_and_merge_configs(self):
        """Load and merge configurations from both nano and eartts checkpoints."""
        print("\n📋 Loading and merging configurations...")
        
        # Load nano's config (for LLM, perception)
        nano_config_file = os.path.join(self.llm_checkpoint_path, "config.json")
        print(f"  Loading nano config: {nano_config_file}")
        with open(nano_config_file, 'r') as f:
            import json
            nano_cfg_dict = json.load(f)
        nano_cfg = DictConfig(nano_cfg_dict)
        
        # Load eartts's config (for TTS)
        eartts_config_file = os.path.join(self.model_path, "config.json")
        print(f"  Loading eartts config: {eartts_config_file}")
        with open(eartts_config_file, 'r') as f:
            eartts_cfg_dict = json.load(f)
        eartts_cfg = DictConfig(eartts_cfg_dict)
        
        # Start with nano's config as base
        merged_cfg = nano_cfg
        
        # Override TTS-related parts with eartts's config
        print("  Merging: Using nano's config for LLM/perception, eartts's for TTS")
        if 'model' in eartts_cfg and 'speech_generation' in eartts_cfg.model:
            merged_cfg.model.speech_generation = eartts_cfg.model.speech_generation
            print("    ✓ TTS config from eartts")
        
        # Set speaker reference
        if 'model' not in merged_cfg:
            merged_cfg.model = {}
        merged_cfg.model.inference_speaker_reference = self.speaker_reference
        
        # Ensure data section has correct sample rates
        if 'data' not in merged_cfg:
            merged_cfg.data = eartts_cfg.data
        
        print(f"  Final config:")
        print(f"    - pretrained_llm: {merged_cfg.model.pretrained_llm}")
        print(f"    - perception.d_model: {merged_cfg.model.perception.modality_adapter.d_model}")
        print(f"    - speech_generation: {'present' if 'speech_generation' in merged_cfg.model else 'missing'}")
        
        return merged_cfg
        
    def _initialize_model(self):
        """Initialize the DuplexS2SExternalSpeechDecoderModel with hybrid loading."""
        from safetensors.torch import load_file
        from nemo.collections.speechlm2.parts.pretrained import set_model_dict_for_partial_init
        
        print("\n🚀 Initializing model with hybrid loading strategy...")
        
        # Step 1: Load and merge configs
        cfg = self._load_and_merge_configs()
        
        # Step 2: DO NOT set pretrained_s2s_model - we'll load weights manually
        cfg.model.pretrained_s2s_model = None
        
        # Convert to dict for model initialization
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        
        # Step 3: Initialize model structure
        print("\n🏗️  Initializing model structure...")
        self.model = DuplexS2SExternalSpeechDecoderModel(cfg_dict)
        print("  ✓ Model structure initialized")
        
        # Step 4: Load nano's checkpoint (LLM + perception)
        if self.llm_checkpoint_path is not None:
            print(f"\n📦 Loading LLM + perception:")
            print(f"  Path: {self.llm_checkpoint_path}")
            
            nano_state_dict = load_file(os.path.join(self.llm_checkpoint_path, "model.safetensors"))
            
            # Filter to non-TTS weights
            tts_keys = ['tts_model.', 'speech_generation.']
            nano_filtered = {k: v for k, v in nano_state_dict.items() 
                           if not any(k.startswith(prefix) for prefix in tts_keys)}
            
            print(f"  Loading {len(nano_filtered)} parameters (excluding TTS)...")
            
            nano_filtered = set_model_dict_for_partial_init(nano_filtered, self.model.state_dict())
            missing, unexpected = self.model.load_state_dict(nano_filtered, strict=False)
            
            missing_non_tts = [k for k in missing if not any(k.startswith(prefix) for prefix in tts_keys)]
            unexpected_non_tts = [k for k in unexpected if not any(k.startswith(prefix) for prefix in tts_keys)]
            
            if missing_non_tts:
                print(f"  ⚠️  {len(missing_non_tts)} non-TTS keys missing (might be OK)")
            if unexpected_non_tts:
                print(f"  ⚠️  {len(unexpected_non_tts)} unexpected non-TTS keys")
        
        # Step 5: Load eartts's checkpoint (TTS only)
        if self.model_path is not None:
            print(f"\n📦 Loading TTS checkpoint:")
            print(f"  Path: {self.model_path}")
            
            eartts_state_dict = load_file(os.path.join(self.model_path, "model.safetensors"))
            
            # Filter to only TTS weights
            tts_keys_filter = ['tts_model.']
            eartts_tts_only = {k: v for k, v in eartts_state_dict.items() 
                                 if any(k.startswith(prefix) for prefix in tts_keys_filter)}
            
            print(f"  Loading {len(eartts_tts_only)} TTS parameters...")
            
            missing, unexpected = self.model.load_state_dict(eartts_tts_only, strict=False)
            
            missing_tts = [k for k in missing if any(k.startswith(prefix) for prefix in tts_keys_filter)]
            unexpected_tts = [k for k in unexpected if any(k.startswith(prefix) for prefix in tts_keys_filter)]
            
            if missing_tts:
                print(f"  ⚠️  {len(missing_tts)} TTS keys missing")
            if unexpected_tts:
                print(f"  ⚠️  {len(unexpected_tts)} unexpected TTS keys")
            
            print(f"  ✓ eartts checkpoint loaded (TTS only)")
        
        print("\n✅ Hybrid loading completed!")
        
        # Setup model
        self.model.to(DEVICE)
        self.model.eval()
        
        # Set working dtype to bfloat16 for S2S part
        self.dtype = torch.bfloat16
        
        # Convert only the S2S components to bfloat16, not the TTS model
        print("Converting S2S components to bfloat16 (keeping TTS in float32)...")
        self.model.llm = self.model.llm.to(torch.bfloat16)
        self.model.lm_head = self.model.lm_head.to(torch.bfloat16)
        self.model.embed_tokens = self.model.embed_tokens.to(torch.bfloat16)
        self.model.perception = self.model.perception.to(torch.bfloat16)
        print("✓ S2S components converted to bfloat16, TTS kept in float32")
        
        self.model.on_train_epoch_start()
        self.tokenizer = self.model.tokenizer
        
        # Get TTS info
        if hasattr(self.model, 'tts_model'):
            self.target_fps = self.model.target_fps
            self.target_sample_rate = self.model.target_sample_rate
            print(f"\nTTS model initialized: target_fps={self.target_fps}, sample_rate={self.target_sample_rate}")
        else:
            print("Warning: TTS model not found in the model")
            
    def _get_bos_embedding(self):
        """Get beginning of sequence embedding."""
        text_bos = torch.full((1,), fill_value=self.model.text_pad_id, device=DEVICE)
        input_embeds = self.model.embed_tokens(text_bos)
        return input_embeds.to(dtype=self.dtype)
    
    def infer_one_step(self,
                       audio_input,
                       num_chunks_to_infer,
                       frame_idx,
                       gen_text,
                       gen_audio,
                       input_embeds_history,
                       dynamic_cache,
                       embedding_position=-1,
                       past_key_values=None,
                       code=None,
                       first_context_subword_id=None,
                       subword_mask=None,
                       generation_config=None,
                       decode_audio=True):
        
        use_cache = dynamic_cache is not None
        batch_size = gen_text.shape[0]
        
        generated_tokens = []
        generated_audio_codes = []
        new_input_embeds = []
        
        for chunk_offset in range(num_chunks_to_infer):
            current_frame_idx = frame_idx + chunk_offset
            
            buffer_len = torch.tensor([audio_input.shape[1]], dtype=torch.long, device=DEVICE)
            source_encoded, _, asr_emb = self.model.perception(
                input_signal=audio_input,
                input_signal_length=buffer_len,
                return_encoder_emb=True,
            )
            source_encoded = source_encoded.to(self.dtype)
            
            current_frame_embedding = source_encoded[:, embedding_position:, :]
            
            current_input_emb = current_frame_embedding.clone()
            current_input_emb *= self.model.cfg.get("duplex_nano_channel_weight", 1.0)
            
            if current_frame_idx == 0:
                current_input_emb += self._get_bos_embedding()
            else:
                last_token_emb = self.model.embed_tokens(gen_text[:, current_frame_idx - 1])
                current_input_emb += last_token_emb
            
            if use_cache:
                if current_frame_idx == 0:
                    ans = self.model(current_input_emb, cache=dynamic_cache)
                else:
                    ans = self.model(current_input_emb, cache=dynamic_cache)
                dynamic_cache = ans["cache"]
            else:
                new_input_embeds.append(current_input_emb)
                full_input_embeds = torch.cat(input_embeds_history + new_input_embeds, dim=1)
                ans = self.model(full_input_embeds, cache=None)
            
            predicted_token = ans["text_logits"][:, -1].argmax(dim=-1)
            gen_text[:, current_frame_idx] = predicted_token
            generated_tokens.append(predicted_token)
            
            if decode_audio and gen_audio is not None and code is not None:
                current_subword_id = gen_text[:, current_frame_idx].unsqueeze(-1)
                
                if self.model.tts_model.cfg.tts_config.context_hidden_size is not None:
                    if current_frame_idx == 0:
                        context_subword_id = first_context_subword_id
                    else:
                        context_subword_id = gen_text[:, current_frame_idx-1].unsqueeze(-1)
                    context_hidden_state = self.model.tts_model.embed_tokens(context_subword_id)
                else:
                    context_hidden_state = None
                
                if self.model.tts_model.cfg.subword_mask_exactly_as_eartts:
                    current_subword_mask = (current_subword_id != self.model.tts_model.text_pad_id).bool()
                else:
                    current_subword_mask = subword_mask[:, current_frame_idx].unsqueeze(-1)
                
                inputs = {
                    "code": code,
                    "context_hidden_state": context_hidden_state,
                    "subword_ids": current_subword_id,
                    "subword_mask": current_subword_mask,
                    "past_key_values": past_key_values,
                    "use_cache": True,
                    "guidance_enabled": True,
                    "generation_config": generation_config,
                    "ignore_eos_flag_stop": True,
                }
                
                outputs = self.model.tts_model.tts_model(**inputs)
                code = outputs.codes
                past_key_values = outputs.past_key_values
                gen_audio[:, current_frame_idx] = code.squeeze(1)
                generated_audio_codes.append(code)
                
                if self.model.cfg.get('inference_force_speech_silence_on_eos', None):
                    silence_codes = self.model.tts_model.codec_silence_tokens.view(1, 1, -1).expand(code.shape)
                    code = torch.where(
                        current_subword_id.unsqueeze(-1) == self.model.tts_model.text_eos_id,
                        silence_codes,
                        code,
                    )
        
        return {
            'gen_text': gen_text,
            'gen_audio': gen_audio,
            'generated_tokens': generated_tokens,
            'generated_audio_codes': generated_audio_codes,
            'input_embeds_history': input_embeds_history + new_input_embeds if not use_cache else input_embeds_history,
            'dynamic_cache': dynamic_cache if use_cache else None,
            'past_key_values': past_key_values,
            'code': code,
        }
    
    def _explore_padding_behavior(self):
        """
        Explore whether the encoder uses left or right padding.
        This helps determine which frame's embedding to use.
        """
        print("\n" + "=" * 70)
        print(" EXPLORING ENCODER PADDING BEHAVIOR")
        print("=" * 70)
        
        # Create test signals with distinctive patterns
        # Test 1: Full buffer
        full_signal = torch.randn(1, self.buffer_size_samples, dtype=self.dtype, device=DEVICE)
        full_len = torch.tensor([self.buffer_size_samples], dtype=torch.long, device=DEVICE)
        
        # Test 2: Half buffer (should be padded)
        half_size = self.buffer_size_samples // 2
        half_signal = torch.randn(1, half_size, dtype=self.dtype, device=DEVICE)
        # Pad to full buffer size
        half_signal_padded = torch.nn.functional.pad(half_signal, (0, self.buffer_size_samples - half_size), value=0.0)
        half_len = torch.tensor([half_size], dtype=torch.long, device=DEVICE)
        
        # Encode both
        with torch.no_grad():
            full_enc, _, _ = self.model.perception(full_signal, full_len, return_encoder_emb=True)
            half_enc, _, _ = self.model.perception(half_signal_padded, half_len, return_encoder_emb=True)
        
        print(f"Full buffer encoding: {full_enc.shape}")
        print(f"Half buffer encoding: {half_enc.shape}")
        
        # Analyze which positions have meaningful embeddings
        full_norm = torch.norm(full_enc, dim=-1)  # [B, T]
        half_norm = torch.norm(half_enc, dim=-1)  # [B, T]
        
        print(f"\nNorm analysis (full buffer): min={full_norm.min():.4f}, max={full_norm.max():.4f}, mean={full_norm.mean():.4f}")
        print(f"Norm analysis (half buffer): min={half_norm.min():.4f}, max={half_norm.max():.4f}, mean={half_norm.mean():.4f}")
        
        # Check first and last few frames
        print(f"\nFirst 3 frame norms (half buffer): {half_norm[0, :3].tolist()}")
        print(f"Last 3 frame norms (half buffer): {half_norm[0, -3:].tolist()}")
        
        # Heuristic: if last frames have significantly higher norm, likely right-padded (valid data at start)
        # if first frames have higher norm, likely left-padded (valid data at end)
        first_avg = half_norm[0, :half_enc.shape[1]//2].mean()
        last_avg = half_norm[0, half_enc.shape[1]//2:].mean()
        
        print(f"\nFirst half average norm: {first_avg:.4f}")
        print(f"Last half average norm: {last_avg:.4f}")
        
        if first_avg > last_avg * 1.2:  # First half significantly larger
            padding_type = "RIGHT"
            useful_position = -1  # Last position corresponds to newest data
            explanation = "Valid data at start, padding at end. New frame info at LAST position."
        elif last_avg > first_avg * 1.2:
            padding_type = "LEFT"
            useful_position = -1  # Still last position, but for different reason
            explanation = "Padding at start, valid data at end. New frame info at LAST position."
        else:
            padding_type = "UNKNOWN (similar norms)"
            useful_position = -1  # Default to last
            explanation = "Cannot determine clearly. Defaulting to LAST position."
        
        print(f"\n{'='*70}")
        print(f"CONCLUSION: Encoder appears to use {padding_type} padding")
        print(f"Explanation: {explanation}")
        print(f"Recommendation: Use embedding at position [{useful_position}] for newest frame")
        print(f"{'='*70}\n")
        
        return useful_position
    
    @torch.no_grad()
    def inference_realtime_streaming(self, audio_path: str, decode_audio: bool = True, 
                                     explore_padding: bool = True):
        """
        Perform realtime streaming inference simulating microphone capture.
        
        Args:
            audio_path: Path to input audio file (simulates microphone input)
            decode_audio: Whether to decode audio codes to waveform
            explore_padding: Whether to explore encoder padding behavior first
            
        Returns:
            Dictionary with 'text', 'tokens_text', 'tokens_audio', 'audio', 'audio_len'
        """
        start_time = time.time()
        
        print("\n" + "=" * 70)
        print("STARTING REALTIME STREAMING INFERENCE")
        print("=" * 70)
        
        # Explore padding behavior if requested
        if explore_padding:
            embedding_position = self._explore_padding_behavior()
        else:
            embedding_position = -1  # Default to last position
            print(f"Using default embedding position: {embedding_position} (last frame)")

        # Load audio file (simulating microphone stream)
        print(f"\n📁 Loading audio file: {audio_path}")
        audio_signal, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
        total_samples = len(audio_signal)
        total_duration = total_samples / SAMPLE_RATE
        total_frames = int(np.ceil(total_samples / FRAME_SIZE_SAMPLES))
        
        print(f"   Total duration: {total_duration:.2f}s")
        print(f"   Total samples: {total_samples}")
        print(f"   Total frames: {total_frames} (each {FRAME_SIZE_SEC}s)")
        
        # Pad audio to exact multiple of frame size
        padded_total_samples = total_frames * FRAME_SIZE_SAMPLES
        if padded_total_samples > total_samples:
            audio_signal = np.pad(audio_signal, (0, padded_total_samples - total_samples), mode='constant')
            print(f"   Padded to: {padded_total_samples} samples")
        
        # Check if Nemotron (no cache support)
        use_cache = 'Nemotron' not in self.model.cfg.pretrained_llm
        print(f"\n⚙️  Model: {self.model.cfg.pretrained_llm}")
        print(f"   Use cache: {use_cache}")
        
        # Initialize buffer and state
        audio_buffer = torch.zeros(1, self.buffer_size_samples, dtype=self.dtype, device=DEVICE)
        buffer_fill_level = 0  # How many samples currently in buffer
        
        # Initialize LLM cache
        if use_cache:
            llm_cache = DynamicCache()
        else:
            llm_cache = None
            input_embeds_history = []  # For no-cache mode
        
        # Initialize TTS
        gen_audio = None
        if decode_audio and hasattr(self.model, 'tts_model'):
            gen_audio = torch.zeros(1, total_frames, self.model.tts_model.tts_model.config.num_quantizers, 
                                   device=DEVICE, dtype=torch.long)
            
            # Load speaker audio
            with fp32_precision():
                speaker_audio, speaker_sr = torchaudio.load(self.speaker_reference)
                speaker_audio = resample(speaker_audio, speaker_sr, self.model.tts_model.target_sample_rate)
            
            speaker_audio = speaker_audio.to(DEVICE)
            speaker_audio_lens = torch.tensor([speaker_audio.size(1)]).long().to(DEVICE)
            
            # Initialize TTS
            init_inputs = self.model.tts_model.get_init_inputs(speaker_audio, speaker_audio_lens, 
                                                                system_prompt=None, user_prompt=None)
            generation_config = self.model.tts_model._get_generation_config(guidance_enabled=True)
            init_inputs.update({"use_cache": True, "past_key_values": None, "guidance_enabled": True})
            
            # Warmup TTS
            outputs = self.model.tts_model.tts_model(**init_inputs)
            code, _, _ = self.model.tts_model.tts_model.generate_step(outputs.hidden_states[:, -1:], **generation_config)
            past_key_values = outputs.past_key_values
            first_context_subword_id = init_inputs["subword_ids"][:, -1].unsqueeze(-1)
            subword_mask = torch.ones(1, total_frames, device=DEVICE, dtype=torch.bool)
            
            print(f"✅ TTS initialized")
        
        gen_text = torch.full((1, total_frames), self.model.text_pad_id, device=DEVICE, dtype=torch.long)
        
        audio_signal_tensor = torch.tensor(audio_signal, dtype=self.dtype, device=DEVICE).unsqueeze(0)
        
        print("\n" + "=" * 70)
        print("🎤 STARTING FRAME-BY-FRAME PROCESSING")
        print("=" * 70)
        
        frame_idx = 0
        while frame_idx < total_frames:
            frame_start = frame_idx * FRAME_SIZE_SAMPLES
            frame_end = frame_start + FRAME_SIZE_SAMPLES
            new_frame = audio_signal_tensor[:, frame_start:frame_end]
            
            if buffer_fill_level < self.buffer_size_samples:
                audio_buffer[:, buffer_fill_level:buffer_fill_level + FRAME_SIZE_SAMPLES] = new_frame
                buffer_fill_level += FRAME_SIZE_SAMPLES
                current_buffer = audio_buffer[:, :buffer_fill_level]
            else:
                audio_buffer = torch.cat([
                    audio_buffer[:, FRAME_SIZE_SAMPLES:],
                    new_frame
                ], dim=1)
                current_buffer = audio_buffer
            
            result = self.infer_one_step(
                audio_input=current_buffer,
                num_chunks_to_infer=1,
                frame_idx=frame_idx,
                gen_text=gen_text,
                gen_audio=gen_audio,
                input_embeds_history=input_embeds_history if not use_cache else [],
                dynamic_cache=llm_cache if use_cache else None,
                embedding_position=embedding_position,
                past_key_values=past_key_values if decode_audio and gen_audio is not None else None,
                code=code if decode_audio and gen_audio is not None else None,
                first_context_subword_id=first_context_subword_id if decode_audio and gen_audio is not None else None,
                subword_mask=subword_mask if decode_audio and gen_audio is not None else None,
                generation_config=generation_config if decode_audio and gen_audio is not None else None,
                decode_audio=decode_audio,
            )
            
            gen_text = result['gen_text']
            gen_audio = result['gen_audio']
            input_embeds_history = result['input_embeds_history']
            llm_cache = result['dynamic_cache']
            if decode_audio and gen_audio is not None:
                past_key_values = result['past_key_values']
                code = result['code']
            
            if frame_idx % 10 == 0 or frame_idx < 3 or gen_text[:, frame_idx].item() == self.model.text_eos_id:
                token_str = self.tokenizer.ids_to_text([gen_text[0, frame_idx].item()])
                buffer_status = f"{buffer_fill_level}/{self.buffer_size_samples}" if buffer_fill_level < self.buffer_size_samples else "FULL"
                special_label = ""
                if gen_text[0, frame_idx].item() == self.model.text_bos_id:
                    special_label = " [BOS]"
                elif gen_text[0, frame_idx].item() == self.model.text_eos_id:
                    special_label = " [EOS]"
                elif gen_text[0, frame_idx].item() == self.model.text_pad_id:
                    special_label = " [PAD]"
                print(f"Frame {frame_idx:3d}/{total_frames} | Buffer: {buffer_status:20s} | Token: {gen_text[0, frame_idx].item():5d}{special_label} | '{token_str}'")
            
            if gen_text[:, frame_idx].item() == self.model.text_eos_id:
                gen_text[:, frame_idx+1:] = self.model.text_pad_id
                total_frames = frame_idx + 1
                break
            
            frame_idx += 1
        
        # Prepare results
        elapsed_time = time.time() - start_time
        print("\n" + "=" * 70)
        print("✅ STREAMING INFERENCE COMPLETED")
        print("=" * 70)
        print(f"Total time: {elapsed_time:.2f}s")
        print(f"Audio duration: {total_duration:.2f}s")
        print(f"RTF (Real-Time Factor): {elapsed_time / total_duration:.2f}x")
        print(f"Processed frames: {total_frames}")
        
        # Trim to actual length
        gen_text = gen_text[:, :total_frames]
        
        # Decode text
        lengths = torch.tensor([total_frames], dtype=torch.long, device=DEVICE)
        text_output = tokens_to_str(gen_text, lengths, tokenizer=self.tokenizer, pad_id=self.model.text_pad_id)
        
        print(f"\n📝 Generated text: {text_output[0][:200]}...")
        
        ans = {
            "text": text_output,
            "tokens_text": gen_text,
            "tokens_len": lengths,
        }
        
        # Decode audio if enabled
        if decode_audio and gen_audio is not None:
            gen_audio = gen_audio[:, :total_frames, :]
            print(f"\n🔊 Decoding audio from {total_frames} frames...")
            gen_audio_codes_lens = torch.tensor([gen_audio.shape[1]]).to(DEVICE)
            
            with fp32_precision(), torch.no_grad():
                audio_pred, audio_len = self.model.tts_model.audio_codec.decode(
                    gen_audio, gen_audio_codes_lens
                )
            
            print(f"   Audio shape: {audio_pred.shape}")
            print(f"   Audio length: {audio_len}")
            
            ans["audio"] = audio_pred.squeeze(1)
            ans["audio_len"] = audio_len
            ans["tokens_audio"] = gen_audio
            print(f"   ✅ Audio decoded successfully")
        
        return ans


def main():
    parser = argparse.ArgumentParser(description="Realtime Streaming Inference (Microphone Simulation)")
    parser.add_argument("--model_path", type=str, 
                       default="/lustre/fsw/portfolios/llmservice/users/cchen1/code/s2s_eartts/Duplex_S2S_Nanov2_30_set_hf/",
                       help="Path to eartts's checkpoint with TTS (HF format)")
    parser.add_argument("--llm_checkpoint_path", type=str,
                       default="/lustre/fsw/portfolios/llmservice/users/cchen1/code/s2s_eartts/nano-9b-model/checkpoints_hf_24002",
                       help="-Path to YOUR checkpoint with correct LLM/perception (HF format)")
    parser.add_argument("--audio_path", type=str, required=True,
                       help="Path to input audio file")
    parser.add_argument("--speaker_reference", type=str,
                       default="/lustre/fsw/portfolios/convai/users/ecasanova/S2S-full-duplex/inference_references/Emma_S3_A1_SC7_singleturntarget_21_channel_1_audio_in.wav",
                       help="Path to speaker reference audio file")
    parser.add_argument("--buffer_size_frames", type=int, default=10,
                       help="Size of audio buffer in frames (each frame = 80ms)")
    parser.add_argument("--output_text", type=str, default="output_text_streaming.txt",
                       help="Output text file path")
    parser.add_argument("--output_audio", type=str, default="generated_audio_streaming.wav",
                       help="Output audio file path")
    parser.add_argument("--decode_audio", action="store_true",
                       help="Whether to decode audio")
    parser.add_argument("--explore_padding", action="store_true", default=False,
                       help="Explore encoder padding behavior (recommended for first run)")
    
    args = parser.parse_args()
    
    try:
        # Initialize model
        print("=" * 70)
        print("REALTIME STREAMING INFERENCE WITH MICROPHONE SIMULATION")
        print("=" * 70)
        print(f"eartts checkpoint (TTS): {args.model_path}")
        print(f"YOUR checkpoint (LLM+perception): {args.llm_checkpoint_path}")
        print(f"Speaker reference: {args.speaker_reference}")
        print(f"Buffer size: {args.buffer_size_frames} frames ({args.buffer_size_frames * FRAME_SIZE_SEC}s)")
        print("=" * 70)
        
        model = RealtimeStreamingInference(
            model_path=args.model_path,
            llm_checkpoint_path=args.llm_checkpoint_path,
            speaker_reference=args.speaker_reference,
            buffer_size_frames=args.buffer_size_frames,
        )
        
        # Run inference
        results = model.inference_realtime_streaming(
            args.audio_path, 
            decode_audio=args.decode_audio,
            explore_padding=args.explore_padding,
        )
        
        # Save outputs
        print("\n" + "=" * 70)
        print("💾 SAVING OUTPUTS")
        print("=" * 70)
        
        # Save text
        with open(args.output_text, 'w') as f:
            f.write(results['text'][0])
        print(f"✅ Text output saved: {args.output_text}")
        
        # Save audio if available
        if 'audio' in results:
            audio_np = results['audio'].cpu().numpy()
            
            import soundfile as sf
            sf.write(args.output_audio, audio_np.flatten(), model.target_sample_rate)
            print(f"✅ Audio output saved: {args.output_audio}")
            
            # Verify
            import subprocess
            size_output = subprocess.check_output(['du', '-h', args.output_audio]).decode().split()[0]
            print(f"   File size: {size_output}")
        
        print("=" * 70)
        print("✅ ALL DONE!")
        print("=" * 70)
        
    except Exception as e:
        print(f"❌ ERROR during inference: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())


