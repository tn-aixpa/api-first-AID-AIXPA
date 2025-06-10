from fastapi import FastAPI, HTTPException, Depends
import uvicorn
import os
# from dialogue_generators_complete import get_complete_generation_options, generate_dialogue_complete
from dialogue_generators_dynamic import  generate_dialogue_dynamic
# from auth import app as auth_app, get_current_active_user, User
import time
import argparse
import json 
from typing import List, Optional, Dict
from pydantic import BaseModel
# from lorax import Client
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

parser = argparse.ArgumentParser()
parser.add_argument('--host', default="0.0.0.0")
parser.add_argument('--port', type=int, default=8013)
parser.add_argument("--lorax_host", default="0.0.0.0")
parser.add_argument("--lorax_port", type=int, default=1234)
parser.add_argument("--hf_token", required = True)
args = parser.parse_args()

lorax_client = base_url="http://" + args.lorax_host + ":" + str(args.lorax_port)

# instantiate FastApi application
app = FastAPI(version="0.0.1")

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=['access-control-allow-origin'],
)

# add authentication
# app.mount("/auth", auth_app)

# add endpoint status
# create_status_endpoint(app)

# create our home page route
@app.get('/')
async def version():
    return {"version": app.version}


# HTTPBearer instance for token checking
security = HTTPBearer()

# Fixed tokens for authentication
FIXED_TOKENS = {
    "w4OqAOToBKiWeHsW3X1zfr0S1rCMY4hA": "interface",
    "029zqQTgWidBqhVMZSpVRFQUTVbbr69v": "test",
}

supported_languages = ["en"]









# Dependency to verify token
async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    print(credentials)
    token = credentials.credentials
    if token in FIXED_TOKENS:
        # Return some information about the user or token if needed
        return {"token": token, "user": FIXED_TOKENS[token]}
    else:
        raise HTTPException(status_code=401, detail="Invalid token")




class DynamicDialogueGenerationRequest(BaseModel):
    generation_mode: str
    documents: List[str]
    dialogue: List[dict]
    user: str
    language: str
    


# API METHODS


@app.post('/dynamic_generation')
def dialogue_generation_dynamic(request: DynamicDialogueGenerationRequest,token: dict = Depends(verify_token)):
    # print(request)
    start_time = time.time()
    print(start_time, "Request dialogue generation")
    documents_list = request.documents
    dialogue_list = request.dialogue
    # print(hf_token)
    # return generate_dialogue_dynamic(request.generation_mode, documents_list, dialogue_list, request.speaker, request.options_number, request.ground_required)
    return generate_dialogue_dynamic(request.generation_mode, documents_list, dialogue_list, request.user, request.language, lorax_client, args.hf_token)

if __name__ == '__main__':

    uvicorn.run("start-api:app", host=args.host, port=args.port, workers=10)
