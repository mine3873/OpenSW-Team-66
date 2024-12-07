import os
import pandas as pd
import json

dataFileName = '02)웰니스_대화_스크립트_데이터셋.xlsx'
project_dir = os.path.dirname(os.path.abspath(__file__))  

file_path = os.path.join(project_dir, dataFileName)

statistics_data = pd.read_excel(file_path, sheet_name="작업 통계")
user_data = pd.read_excel(file_path, sheet_name="사용자 발화")

statistics_columns = ["중분류", "시작행", "끝행"]
statistics_data = statistics_data[statistics_columns]
# 중분류 : 사용자의 감정 상태
# 시작행 : 해당 중분류 데이터의 첫 행
# 끝 행  : 해당 중분류 데이터의 마지막 행

columns_to_use = ["intent", "utterance", "utterance(2차)", "response(공감)"]
user_data_filtered = user_data[columns_to_use]
# intent : 중분류와 같음
# utterance, utterance(2차) : 사용자 발화
# response(공감) : 사용자 발화에 대한 챗봇의 답변 데이터


def create_json(statistics_data, user_data_filtered):
    json_data = {}

    # 작업 통계시트에서 각 중분류의 시작행과 끝행 파악 
    for _, row in statistics_data.iterrows():
        intent = row["중분류"] 

        # 해당 중분류에 대한 시작행, 끝행 값이 NaN인 경우가 있음. 이 경우 continue
        if pd.isna(row["시작행"]) or pd.isna(row["끝행"]):
            continue

        start_row = int(row["시작행"]) - 1  
        end_row = int(row["끝행"]) - 1
        # Pandas는 인덱스가 0부터 시작, .xlsx의 경우 1부터 시작 
        
        #각 intent에 대한 사용자 발화 데이터 추출
        intent_data = user_data_filtered.iloc[start_row:end_row + 1]

        #각 intent에 대한 챗봇 답변 데이터 추출
        responses = intent_data["response(공감)"].dropna().tolist()

        
        prompts = []
        for _, user_row in intent_data.iterrows():
            if pd.notna(user_row["utterance"]):
                prompts.append({
                    "prompt": user_row["utterance"],
                    "response": responses[len(prompts) % len(responses)]
                })
            if pd.notna(user_row["utterance(2차)"]):
                prompts.append({
                    "prompt": user_row["utterance(2차)"],
                    "response": responses[len(prompts) % len(responses)]
                })
        #사용자 발화 데이터 양에 비해 챗봇 답변 데이터의 수가 적음
        #(intent 데이터의 수) mod (답변 데이터의 수) 를 통해 각 
        #사용자 발화에 의한 답변 데이터 설정
        
        json_data[intent] = prompts
    return json_data


json_result = create_json(statistics_data, user_data_filtered)

output_file = project_dir + "/Data_ForChatGPT.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(json_result, f, ensure_ascii=False, indent=4)

print(f"JSON 데이터 저장 : {output_file}")
# 데이터 참고 링크
# https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=267