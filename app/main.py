import os
from fastapi import FastAPI
from huggingface_hub import login
from app.router.v1.base import router as route_v1
from app.configs.environment import get_environment_variables

env = get_environment_variables()

app = FastAPI(
  title = env.APP_NAME,
) 

@app.on_event("startup")
def startup_event():
    hf_token = env.HUGGINGFACE_ACCESS_TOKEN
    if not hf_token:
        raise RuntimeError("HuggingFace access token not found in environment.")
    login(token=hf_token)

app.include_router(route_v1, prefix='/api/v1')