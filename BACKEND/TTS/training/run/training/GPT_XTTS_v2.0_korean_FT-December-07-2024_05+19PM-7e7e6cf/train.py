import os

from trainer import Trainer, TrainerArgs
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainer, GPTTrainerConfig, XttsAudioConfig
from TTS.utils.manage import ModelManager

RUN_NAME = "GPT_XTTS_v2.0_korean_FT"
PROJECT_NAME = "XTTS_trainer"
DASHBOARD_LOGGER = "tensorboard"
LOGGER_URI = None

OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "BACKEND", "TTS", "training" ,"run", "training")
os.makedirs(OUT_PATH, exist_ok=True)

OPTIMIZER_WD_ONLY_ON_WEIGHTS = True
START_WITH_EVAL = True
BATCH_SIZE = 3
GRAD_ACUMM_STEPS = 84

config_dataset = BaseDatasetConfig(
    formatter="kss", 
    dataset_name="korean_dataset",
    path="BACKEND/TTS/training/run/training\wav_resampled",
    meta_file_train="transcript.txt",
    language="ko",
)

DATASETS_CONFIG_LIST = [config_dataset]

CHECKPOINTS_OUT_PATH = os.path.join(OUT_PATH, "XTTS_v2.0_original_model_files/")
os.makedirs(CHECKPOINTS_OUT_PATH, exist_ok=True)

DVAE_CHECKPOINT_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/dvae.pth"
MEL_NORM_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/mel_stats.pth"
XTTS_CHECKPOINT_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/model.pth"
TOKENIZER_FILE_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/vocab.json"

DVAE_CHECKPOINT = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(DVAE_CHECKPOINT_LINK))
MEL_NORM_FILE = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(MEL_NORM_LINK))
XTTS_CHECKPOINT = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(XTTS_CHECKPOINT_LINK))
TOKENIZER_FILE = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(TOKENIZER_FILE_LINK))

if not os.path.isfile(DVAE_CHECKPOINT) or not os.path.isfile(MEL_NORM_FILE):
    ModelManager._download_model_files([MEL_NORM_LINK, DVAE_CHECKPOINT_LINK], CHECKPOINTS_OUT_PATH, progress_bar=True)

if not os.path.isfile(TOKENIZER_FILE) or not os.path.isfile(XTTS_CHECKPOINT):
    ModelManager._download_model_files([TOKENIZER_FILE_LINK, XTTS_CHECKPOINT_LINK], CHECKPOINTS_OUT_PATH, progress_bar=True)

SPEAKER_REFERENCE = ["BACKEND/TTS/training/run/training/wav_resampled/1/1_0000.wav", "BACKEND/TTS/training/run/training/wav_resampled/1/1_0001.wav"]
LANGUAGE = "ko"

test_sentences = [
    {"text": "안녕하세요, 오늘 날씨가 참 좋네요.", "speaker_wav": SPEAKER_REFERENCE, "language": LANGUAGE},
    {"text": "한국어 음성 합성을 학습 중입니다.", "speaker_wav": SPEAKER_REFERENCE, "language": LANGUAGE},
    {"text": "오픈소스는 재밌습니다.", "speaker_wav": SPEAKER_REFERENCE, "language": LANGUAGE},
    {"text": "다음 주가 시험입니다.", "speaker_wav": SPEAKER_REFERENCE, "language": LANGUAGE},
    {"text": "그만 쉬고 싶습니다.", "speaker_wav": SPEAKER_REFERENCE, "language": LANGUAGE}
]

model_args = GPTArgs(
    max_conditioning_length=132300,
    min_conditioning_length=66150,
    max_wav_length=255995,
    max_text_length=200,
    mel_norm_file=MEL_NORM_FILE,
    dvae_checkpoint=DVAE_CHECKPOINT,
    xtts_checkpoint=XTTS_CHECKPOINT,
    tokenizer_file=TOKENIZER_FILE,
    gpt_num_audio_tokens=1026,
    gpt_start_audio_token=1024,
    gpt_stop_audio_token=1025,
    gpt_use_masking_gt_prompt_approach=True,
    gpt_use_perceiver_resampler=True,
)

audio_config = XttsAudioConfig(
    sample_rate=22050,
    dvae_sample_rate=22050,
    output_sample_rate=24000
)

config = GPTTrainerConfig(
    output_path=OUT_PATH,
    model_args=model_args,
    run_name=RUN_NAME,
    project_name=PROJECT_NAME,
    batch_size=BATCH_SIZE,
    optimizer="AdamW",
    optimizer_params={"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": 1e-2},
    #lr=5e-6,
    lr = 1e-4,
    test_sentences=test_sentences,
    audio=audio_config 
)

def main():
    model = GPTTrainer.init_from_config(config)
    train_samples, eval_samples = load_tts_samples(
        DATASETS_CONFIG_LIST,
        eval_split=True,
        eval_split_max_size=256,
    )
    trainer = Trainer(
        TrainerArgs(
            skip_train_epoch=False,
            start_with_eval=START_WITH_EVAL,
            grad_accum_steps=GRAD_ACUMM_STEPS, 
        ),
        config,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )
    trainer.fit()

if __name__ == "__main__":
    main()
