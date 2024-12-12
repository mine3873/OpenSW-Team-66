import speech_recognition as sr

class SpeechToText:
    def __init__(self):
        self.recognizer = sr.Recognizer()

    def recognize_speech(self):
        """마이크 입력을 텍스트로 변환합니다."""
        with sr.Microphone() as source:
            print("\nSpeak now... (Listening)")
            try:
                
                audio = self.recognizer.listen(source, timeout=10)
                text = self.recognizer.recognize_google(audio, language="ko-KR")
                print(f"Recognized Speech: {text}")
                return text
            except sr.UnknownValueError:
                print("Sorry, I couldn't understand the speech. Please try again.")
                return None
            except sr.RequestError as e:
                print(f"STT Error: {e}")
                return None
