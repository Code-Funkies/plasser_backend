from fastapi import FastAPI
from inference_service import inference_service

app = FastAPI()


@app.get("/api/inference")
async def inference():
    return inference_service()

@app.get("/")
async def root():
    return {"message": "Hello World"}
