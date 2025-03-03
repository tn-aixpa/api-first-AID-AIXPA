import requests
from jprint import jprint
from pydantic import BaseModel
import time
import json
from typing import List, Optional, Dict




class DynamicDialogueGenerationRequest(BaseModel):
    generation_mode: str
    documents: List[str]
    dialogue: List[dict]
    user: str
    language: str
    
    

##################################################
##################################################
#################### DYNAMIC #####################
##################################################
##################################################

request_dynamic = DynamicDialogueGenerationRequest(
    generation_mode="aixpa_chatbot_lorax",
    documents= ['''PIANO FAMIGLIA DI MEZZOCORONA ANNO 2023 '''],
    dialogue= [
        {
            "speaker": "speaker_2",
            "turn_text": "Buongiorno, come posso essere utile?"
        },
        {
            "speaker": "speaker_1",
            "turn_text": "Vorrei capire quali azioni posso proporre in un piano famiglia prendendo spunto dal piano di Mezzocorona"
        }  
    ],
    user="operatore",
    language="informal"


) 
# print(request_dynamic.json)
endpoint_url = "http://localhost:8013/dynamic_generation/"

headers = {"Authorization": "Bearer w4OqAOToBKiWeHsW3X1zfr0S1rCMY4hA"}

response = requests.post(endpoint_url, request_dynamic.json(), headers=headers)
print(response.status_code)
# jprint(response.json()) 
print(json.dumps(response.json(), indent=4, ensure_ascii=False))

