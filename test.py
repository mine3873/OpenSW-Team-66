
from BACKEND.TTS.TTS.tts.utils.text.tokenizer import TTSTokenizer
# 설정
vocab_file = "vocab.txt"
config = {"use_phonemes": False, "vocab_file": vocab_file}
tokenizer = TTSTokenizer(config=config)

# 테스트
text = "안녕하세요. 이것은 테스트입니다."
tokens = tokenizer.text_to_sequence(text)
print(f"토큰화 결과: {tokens}")
