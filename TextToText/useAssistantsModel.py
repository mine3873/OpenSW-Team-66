import os
from openai import OpenAI, AssistantEventHandler
from typing_extensions import override
from dotenv import load_dotenv

load_dotenv(dotenv_path="TextToText\src\API key\APIKEY.env")

#OPENAI 객체
client = OpenAI(
    api_key=os.getenv('OPENAI_API_KEY') 
)

# assistants GPT model 로드
assistant = client.beta.assistants.retrieve(
    assistant_id=os.getenv('ASSISTANT_ID')  
)

thread = client.beta.threads.create()

class EventHandler(AssistantEventHandler):
    @override
    def on_text_created(self, text) -> None:
        print(f"\n어시스턴트: ", end="", flush=True)

    #텍스트를 점진적으로 출력
    @override
    def on_text_delta(self, delta, snapshot):
        print(delta.value, end="", flush=True)

def ask_assistant_streaming(question):
    try:
        with client.beta.threads.runs.stream(
            thread_id=thread.id,
            assistant_id=assistant.id,  
            instructions="대화형 심리 상담 모델입니다.",  
            additional_messages=[{"role": "user", "content": question}],  
            event_handler=EventHandler(),
            max_completion_tokens=300,
            temperature=0.7,  
            top_p=0.9,         
        ) as stream:
            stream.until_done()
    except Exception as e:
        print(f"오류 발생: {e}")

# 대화 루프
if __name__ == "__main__":
    print("어시스턴트 테스트 (종료하려면 'exit' 입력)")
    while True:
        user_input = input("사용자: ")
        if user_input.lower() == "exit":
            print("대화 종료")
            break
        ask_assistant_streaming(user_input)
        print()
