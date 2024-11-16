from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware 
import os
from groq_services import GroqServices
from career_service import CareerAdviceRequest, get_career_advice
from general_service import GeneralRequest, get_general_response
from agent_router import Router, RouterRequest
from pydantic import BaseModel
from pathlib import Path


parent_directory = os.path.dirname(os.getcwd())
AUDIO_DIR = f"{parent_directory}/audio-files"

app = FastAPI()
router = Router()

# Configuration
MAX_FILE_SIZE = 25 * 1024 * 1024  # 25MB


origins = ["http://localhost:3000", "http://127.0.0.1:8000"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str


@app.post("/chat")
async def chat(request: ChatRequest):
    # Create router request from chat request
    router_request = RouterRequest(message=request.message)
    
    # Route the request to the appropriate agent
    agent_type, reason = router.route(router_request)
    
    # Then, get response from the appropriate agent
    if agent_type == "career_agent":
        career_request = CareerAdviceRequest(prompt=request.message)
        return StreamingResponse(
            get_career_advice(career_request),
            media_type="text/event-stream"
        )
    else:  # Default to general agent
        general_request = GeneralRequest(prompt=request.message)
        return StreamingResponse(
            get_general_response(general_request),
            media_type="text/event-stream"
        )


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}


@app.post("/transcript/")
async def upload_audio(file: UploadFile = File(...)):
    save_dir = Path("uploaded_audio")
    save_dir.mkdir(exist_ok=True)

    try:
        if not file:
            return JSONResponse(
                status_code=400,
                content={"error": "No file uploaded"}
            )

        file_path = save_dir / file.filename
        with file_path.open("wb") as audio_file:
            audio_file.write(await file.read())  

        filename = os.path.dirname(__file__) + f"/{file_path}"
        transcription = GroqServices().speech_to_text(filename)
        os.remove(filename)

        return JSONResponse(
            status_code=200,
            content={
                "transcription": transcription,
            }
        )
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"An error occurred: {str(e)}"}
        )
