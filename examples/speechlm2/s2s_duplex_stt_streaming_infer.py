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

cache_dir="/lustre/fsw/portfolios/llmservice/users/kevinhu/hfcache"
os.environ["HF_HOME"] = cache_dir
os.environ["TORCH_HOME"] = cache_dir
os.environ["NEMO_CACHE_DIR"] = cache_dir

from nemo.collections.speechlm2.models.duplex_stt_model import DuplexSTTModel, tokens_to_str

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SAMPLE_RATE = 16000
FRAME_SIZE_SEC = 0.08
FRAME_SIZE_SAMPLES = int(SAMPLE_RATE * FRAME_SIZE_SEC)


class RealtimeStreamingSTTInference:
    """
    Realtime streaming STT inference that simulates microphone data capture.
    Uses a sliding window buffer and processes audio frame by frame.
    Text-only output (no audio generation).
    """
    
    def __init__(self, 
                 model_path: str, 
                 buffer_size_frames: int = 10):
        """
        Initialize the model for realtime streaming inference.
        
        Args:
            model_path (str): Path to checkpoint (HF format)
            buffer_size_frames (int): Size of audio buffer in frames (each frame = 80ms)
        """
        print("=" * 70)
        print("INITIALIZING REALTIME STREAMING STT INFERENCE")
        print("=" * 70)
        print(f"Frame size: {FRAME_SIZE_SEC}s ({FRAME_SIZE_SAMPLES} samples @ {SAMPLE_RATE}Hz)")
        print(f"Buffer size: {buffer_size_frames} frames ({buffer_size_frames * FRAME_SIZE_SEC}s)")
        print("=" * 70)
        
        self.model_path = model_path
        self.buffer_size_frames = buffer_size_frames
        self.buffer_size_samples = buffer_size_frames * FRAME_SIZE_SAMPLES
        
        self.model = None
        self.tokenizer = None
        self.dtype = None
        
        self._initialize_model()
        
        print(f"\n✅ RealtimeStreamingSTTInference initialized successfully.")
        
    def _initialize_model(self):
        """Initialize the DuplexSTTModel."""
        from safetensors.torch import load_file
        
        print(f"\n🚀 Initializing DuplexSTTModel...")
        print(f"  Loading from: {self.model_path}")
        
        config_file = os.path.join(self.model_path, "config.json")
        with open(config_file, 'r') as f:
            import json
            cfg_dict = json.load(f)
        
        cfg = DictConfig(cfg_dict)
        cfg.model.pretrained_weights = False
        
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        
        print("\n🏗️  Initializing model structure...")
        self.model = DuplexSTTModel(cfg_dict)
        print("  ✓ Model structure initialized")
        
        print(f"\n📦 Loading checkpoint weights...")
        state_dict = load_file(os.path.join(self.model_path, "model.safetensors"))
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        
        if missing:
            print(f"  ⚠️  {len(missing)} keys missing")
        if unexpected:
            print(f"  ⚠️  {len(unexpected)} unexpected keys")
        
        print(f"  ✓ Checkpoint loaded")
        
        self.model.to(DEVICE)
        self.model.eval()
        
        self.dtype = torch.bfloat16
        
        print("Converting model components to bfloat16...")
        self.model.llm = self.model.llm.to(torch.bfloat16)
        self.model.lm_head = self.model.lm_head.to(torch.bfloat16)
        self.model.embed_tokens = self.model.embed_tokens.to(torch.bfloat16)
        self.model.perception = self.model.perception.to(torch.bfloat16)
        if self.model.predict_user_text:
            self.model.embed_asr_tokens = self.model.embed_asr_tokens.to(torch.bfloat16)
        print("✓ Model components converted to bfloat16")
        
        self.model.on_train_epoch_start()
        self.tokenizer = self.model.tokenizer
        
        print(f"\nModel info:")
        print(f"  - LLM: {self.model.cfg.pretrained_llm}")
        print(f"  - Predict user text (ASR): {self.model.predict_user_text}")
        
    def _explore_padding_behavior(self):
        """
        Explore whether the encoder uses left or right padding.
        This helps determine which frame's embedding to use.
        """
        print("\n" + "=" * 70)
        print(" EXPLORING ENCODER PADDING BEHAVIOR")
        print("=" * 70)
        
        full_signal = torch.randn(1, self.buffer_size_samples, dtype=self.dtype, device=DEVICE)
        full_len = torch.tensor([self.buffer_size_samples], dtype=torch.long, device=DEVICE)
        
        half_size = self.buffer_size_samples // 2
        half_signal = torch.randn(1, half_size, dtype=self.dtype, device=DEVICE)
        half_signal_padded = torch.nn.functional.pad(half_signal, (0, self.buffer_size_samples - half_size), value=0.0)
        half_len = torch.tensor([half_size], dtype=torch.long, device=DEVICE)
        
        with torch.no_grad():
            full_enc, _, _ = self.model.perception(full_signal, full_len, return_encoder_emb=True)
            half_enc, _, _ = self.model.perception(half_signal_padded, half_len, return_encoder_emb=True)
        
        print(f"Full buffer encoding: {full_enc.shape}")
        print(f"Half buffer encoding: {half_enc.shape}")
        
        full_norm = torch.norm(full_enc, dim=-1)
        half_norm = torch.norm(half_enc, dim=-1)
        
        print(f"\nNorm analysis (full buffer): min={full_norm.min():.4f}, max={full_norm.max():.4f}, mean={full_norm.mean():.4f}")
        print(f"Norm analysis (half buffer): min={half_norm.min():.4f}, max={half_norm.max():.4f}, mean={half_norm.mean():.4f}")
        
        print(f"\nFirst 3 frame norms (half buffer): {half_norm[0, :3].tolist()}")
        print(f"Last 3 frame norms (half buffer): {half_norm[0, -3:].tolist()}")
        
        first_avg = half_norm[0, :half_enc.shape[1]//2].mean()
        last_avg = half_norm[0, half_enc.shape[1]//2:].mean()
        
        print(f"\nFirst half average norm: {first_avg:.4f}")
        print(f"Last half average norm: {last_avg:.4f}")
        
        if first_avg > last_avg * 1.2:
            padding_type = "RIGHT"
            useful_position = -1
            explanation = "Valid data at start, padding at end. New frame info at LAST position."
        elif last_avg > first_avg * 1.2:
            padding_type = "LEFT"
            useful_position = -1
            explanation = "Padding at start, valid data at end. New frame info at LAST position."
        else:
            padding_type = "UNKNOWN (similar norms)"
            useful_position = -1
            explanation = "Cannot determine clearly. Defaulting to LAST position."
        
        print(f"\n{'='*70}")
        print(f"CONCLUSION: Encoder appears to use {padding_type} padding")
        print(f"Explanation: {explanation}")
        print(f"Recommendation: Use embedding at position [{useful_position}] for newest frame")
        print(f"{'='*70}\n")
        
        return useful_position
    
    @torch.no_grad()
    def inference_realtime_streaming(self, audio_path: str, explore_padding: bool = True):
        """
        Perform realtime streaming inference simulating microphone capture.
        Uses the model's built-in streaming_inference method.
        
        Args:
            audio_path: Path to input audio file (simulates microphone input)
            explore_padding: Whether to explore encoder padding behavior first
            
        Returns:
            Dictionary with 'text', 'tokens_text', 'tokens_len', and optionally 'src_text', 'tokens_text_src'
        """
        start_time = time.time()
        
        print("\n" + "=" * 70)
        print("STARTING REALTIME STREAMING STT INFERENCE")
        print("=" * 70)
        
        if explore_padding:
            embedding_position = self._explore_padding_behavior()
        else:
            embedding_position = -1
            print(f"Using default embedding position: {embedding_position} (last frame)")

        print(f"\n📁 Loading audio file: {audio_path}")
        audio_signal, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
        total_samples = len(audio_signal)
        total_duration = total_samples / SAMPLE_RATE
        
        print(f"   Total duration: {total_duration:.2f}s")
        print(f"   Total samples: {total_samples}")
        
        audio_signal_tensor = torch.tensor(audio_signal, dtype=self.dtype, device=DEVICE).unsqueeze(0)
        
        print(f"\n⚙️  Model: {self.model.cfg.pretrained_llm}")
        print(f"   Buffer size: {self.buffer_size_frames} frames")
        
        print("\n" + "=" * 70)
        print("🎤 STARTING STREAMING INFERENCE (using model.streaming_inference)")
        print("=" * 70)
        
        results = self.model.streaming_inference(
            audio_signal=audio_signal_tensor,
            frame_size_samples=FRAME_SIZE_SAMPLES,
            buffer_size_frames=self.buffer_size_frames,
            embedding_position=embedding_position,
        )
        
        elapsed_time = time.time() - start_time
        print("\n" + "=" * 70)
        print("✅ STREAMING INFERENCE COMPLETED")
        print("=" * 70)
        print(f"Total time: {elapsed_time:.2f}s")
        print(f"Audio duration: {total_duration:.2f}s")
        print(f"RTF (Real-Time Factor): {elapsed_time / total_duration:.2f}x")
        print(f"Processed frames: {results['tokens_len'].item()}")
        
        print(f"\n📝 Generated text: {results['text'][0]}")
        
        if results['src_text'] is not None:
            print(f"📝 Source text (ASR): {results['src_text'][0]}")
        
        return results


def main():
    parser = argparse.ArgumentParser(description="Realtime Streaming STT Inference (Text-Only)")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to checkpoint (HF format)")
    parser.add_argument("--audio_path", type=str, required=True,
                       help="Path to input audio file")
    parser.add_argument("--buffer_size_frames", type=int, default=10,
                       help="Size of audio buffer in frames (each frame = 80ms)")
    parser.add_argument("--output_text", type=str, default="output_text_streaming_stt.txt",
                       help="Output text file path")
    parser.add_argument("--explore_padding", action="store_true", default=False,
                       help="Explore encoder padding behavior (recommended for first run)")
    
    args = parser.parse_args()
    
    try:
        print("=" * 70)
        print("REALTIME STREAMING STT INFERENCE (TEXT-ONLY)")
        print("=" * 70)
        print(f"Model checkpoint: {args.model_path}")
        print(f"Audio file: {args.audio_path}")
        print(f"Buffer size: {args.buffer_size_frames} frames ({args.buffer_size_frames * FRAME_SIZE_SEC}s)")
        print("=" * 70)
        
        model = RealtimeStreamingSTTInference(
            model_path=args.model_path,
            buffer_size_frames=args.buffer_size_frames,
        )
        
        results = model.inference_realtime_streaming(
            args.audio_path, 
            explore_padding=args.explore_padding,
        )
        
        print("\n" + "=" * 70)
        print("💾 SAVING OUTPUTS")
        print("=" * 70)
        
        with open(args.output_text, 'w') as f:
            f.write(results['text'][0])
        print(f"✅ Text output saved: {args.output_text}")
        
        if 'src_text' in results:
            src_output_path = args.output_text.replace('.txt', '_src.txt')
            with open(src_output_path, 'w') as f:
                f.write(results['src_text'][0])
            print(f"✅ Source text output saved: {src_output_path}")
        
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

