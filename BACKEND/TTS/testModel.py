import os
import torch
import torchaudio
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

print("Loading model...")
config = XttsConfig()
config.load_json("BACKEND/TTS/training/run/training/GPT_XTTS_v2.0_korean_FT-December-07-2024_05+19PM-7e7e6cf/config.json")
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_dir="BACKEND/TTS/training/run/training/GPT_XTTS_v2.0_korean_FT-December-07-2024_05+19PM-7e7e6cf", use_deepspeed=False)
model.cuda()

print("Computing speaker latents...")
gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=["BACKEND/TTS/training/run/training/wav_resampled/1/1_0000.wav"])

print("Inference...")
out = model.inference(
    "저는 심리상담 전문 챗봇 입니다. 반갑습니다",
    "ko",
    gpt_cond_latent,
    speaker_embedding,
    temperature=0.7, # Add custom parameters here
)
torchaudio.save("BACKEND/TTS/sample_Generated_Audio/step_8484/Audio.wav", torch.tensor(out["wav"]).unsqueeze(0), 24000)