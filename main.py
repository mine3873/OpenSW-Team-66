import os
import threading
import warnings
import logging
import time
from openai import OpenAI, AssistantEventHandler
from typing_extensions import override
from dotenv import load_dotenv
import torch
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import sounddevice as sd
import numpy as np
import concurrent.futures

CONFIG_PATH = "BACKEND/TTS/training/run/training/GPT_XTTS_v2.0_korean_FT-December-07-2024_05+19PM-7e7e6cf/config.json"
# config.json 의 경로
MODEL_FILE_PATH = "BACKEND/TTS/training/run/training/GPT_XTTS_v2.0_korean_FT-December-07-2024_05+19PM-7e7e6cf"
# model.pth 파일이 존재하는 파일 경로

API_KEY_PATH = "BACKEND/textGenerate/src/API key/APIKEY.env"
# APIKEY.env 파일 경로

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
# 로그 출력 최소

load_dotenv(dotenv_path=API_KEY_PATH)

def load_openai():
    global client, assistant, thread
    client = OpenAI(
        api_key=os.getenv('OPENAI_API_KEY') 
    )
    assistant = client.beta.assistants.retrieve(
        assistant_id=os.getenv('ASSISTANT_ID')  
    )
    thread = client.beta.threads.create()

def load_tts():
    global model, gpt_cond_latent, speaker_embedding
    config = XttsConfig()
    config.load_json(CONFIG_PATH)
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_dir=MODEL_FILE_PATH, use_deepspeed=False)
    model.cuda()
    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=["BACKEND/TTS/training/run/training/wav_resampled/1/1_0000.wav"])

# openai, TTS 모델 로드
def load_both():
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(load_openai),
            executor.submit(load_tts),
        ]
        concurrent.futures.wait(futures)

class EventHandler(AssistantEventHandler):
    def __init__(self):
        super().__init__()
        self.generated_text = "" # 생성된 답변 

    @override
    def on_text_created(self, text) -> None:
        print(f"어시스턴트: ", end="", flush=True)

    @override
    def on_text_delta(self, delta, snapshot):
        self.generated_text += str(delta.value)

# 질문에 대한 답변 생성 및 리턴 
def ask_assistant_streaming(question):
    handler = EventHandler()
    try:
        with client.beta.threads.runs.stream(
            thread_id=thread.id,
            assistant_id=assistant.id,  
            instructions="대화형 심리 상담 모델입니다. 응답은 최대 80자를 초과하지 않도록 간결하고 핵심내용만 전달해주세요.",  
            additional_messages=[{"role": "user", "content": question}],  
            event_handler=handler,
            max_completion_tokens=100,
            temperature=0.4,  
            top_p=0.9,         
            # 파라미터 값 조정 필요
        ) as stream:
            stream.until_done()
        return handler.generated_text

    except Exception as e:
        print(f"오류 발생: {e}")


volume_factor = 0.3
# 생성된 답변은 문장 단위로 쪼갬 -> 긴 문장에 대해 TTS 음성 생성 시, 오류 발생 확률 UP
def split_text_into_sentences(text):
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return sentences

# 답변에 대한 TTS 음성 생성 및 재생
def createTTS(response):
    try:
        sentences = split_text_into_sentences(response)
        audio_data_list = []

        silence_duration = 0.3 
        silence = np.zeros(int(24000 * silence_duration))

        for idx, sentence in enumerate(sentences):
            out = model.inference(
                sentence,
                "ko",
                gpt_cond_latent,
                speaker_embedding,
                temperature=0.7,
            )
            audio_data = torch.tensor(out["wav"]).numpy()
            audio_data_list.append(audio_data * volume_factor)
            audio_data_list.append(silence)  
            
        full_audio = np.concatenate(audio_data_list)
        sd.play(full_audio, samplerate=24000)
    except Exception as e:
        print(f"TTS 생성 오류: {e}")

# 답변 생성 및 TTS 음성 생성을 병렬로 처리, 
def playWithThread(question):
    def create_answer_and_TTS_audio():
        response = ask_assistant_streaming(question)
        createTTS(response)
        for i in response:
            print(i, end='', flush=True)
            time.sleep(0.05)
    tts_thread = threading.Thread(target=create_answer_and_TTS_audio)
    tts_thread.start()
    tts_thread.join() 

# 대화 루프
if __name__ == "__main__":
    load_both()
    print("어시스턴트 테스트 (종료하려면 'exit' 입력)")
    while True:
        user_input = input("사용자: ")
        if user_input.lower() == "exit":
            print("대화 종료")
            break
        playWithThread(user_input)
        print()
