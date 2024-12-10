import sounddevice as sd
import wave

AUDIO_PATH = "BACKEND/TTS/sample_Generated_Audio/Audio.wav"

def play_wav_file(file_path):
    try:
        with wave.open(file_path, 'rb') as wf:
            # 오디오 파일 정보 가져오기
            sample_rate = wf.getframerate()
            channels = wf.getnchannels()
            frames = wf.readframes(wf.getnframes())

            # WAV 데이터를 재생
            sd.play(frames, samplerate=sample_rate, channels=channels)
            sd.wait()  # 재생이 끝날 때까지 대기
    except Exception as e:
        print(f"오디오 재생 오류 발생: {e}")

play_wav_file(AUDIO_PATH)
