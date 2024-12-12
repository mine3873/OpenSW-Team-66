# OpenSW-Team-66

## introduction 
We want to create the chatBot aimed at psychological counseling. 
So using open-ai and TTS open source, we started this project.

``` cmd
git clone https://github.com/mine3873/OpenSW-Team-66.git
```


## install
```python ver==3.10```

``` cmd
pip install -r requirements.txt
```

You must create APIKEY.env for this program.
the file cosists of 
``` markdown
OPENAI_API_KEY= YOUR_API_KEY_FROM_OPEN_AI
```
you must create API key from open-ai to run 
--> [https://platform.openai.com/docs/overview](https://platform.openai.com/docs/overview)


## DOWNLOAD trained model.pth
[download](https://1drv.ms/u/s!AmtfKlFp1bieg41RKB_MvACAbzrDHQ?embed=1) model.pth and save into 
--> `BACKEND/TTS/training/run/training/TTSMODEL`


## RUN
``` python
API_KEY_PATH = "YOUR OPEN AI APIKEY.env FILE PATH"
# APIKEY.env 파일 경로
```
if you created open-ai API KEY and APIKEY.env, insert the PATH of APIKEY.env into `API_KEY_PATH`

then, try below in terminal. 
``` cmd
python main.py
```

### SAMPLE VIDEO
[Watch the test video](https://raw.githubusercontent.com/mine3873/OpenSW-Team-66/master/BACKEND/src/video/test.mp4)

## references
- How to fine tuning TTS model
    - [https://www.kaggle.com/code/maxbr0wn/fine-tuning-xtts-v2-english](https://www.kaggle.com/code/maxbr0wn/fine-tuning-xtts-v2-english)

- DataSet for fine tuning
    - [[https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=267](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=267)]

- How to use ASSISTANT model 
    - [https://platform.openai.com/docs/assistants/quickstart](https://platform.openai.com/docs/assistants/quickstart)


test