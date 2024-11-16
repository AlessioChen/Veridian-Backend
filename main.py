from llm_service import LLMService, ChatRequest
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from groq_services import GroqServices
import os
import json
from perplexity_service import PerplexityService
from models.user_profile import UserProfile

app = FastAPI()
llm_service = LLMService()

# Configuration
MAX_FILE_SIZE = 25 * 1024 * 1024  # 25MB


origins = ["http://localhost:3000", "http://127.0.0.1:8000", "http://localhost:8000", "http://127.0.0.1:5500", "http://localhost:5500"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/chat")
async def chat(request: ChatRequest):
    print(f"Chat endpoint received request: {request.message}")
    
    async def generate_response():
        print("Starting response generation in endpoint")
        try:
            async for chunk in llm_service.generate_response("default_user", request.message):
                content = chunk.get('content', '')
                if content:
                    # Format as SSE
                    yield f"data: {json.dumps({'content': content})}\n\n"
        except Exception as e:
            print(f"Error in endpoint generate_response: {str(e)}")
            raise

    return StreamingResponse(
        generate_response(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )

@app.post("/transcript/")
async def upload_audio(file: UploadFile = File(...)):
    SAVE_DIR = Path("uploaded_audio")
    SAVE_DIR.mkdir(exist_ok=True)

@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}

@app.post("/user-profile")
async def create_profile(user_profile: UserProfile):
    x = user_profile

@app.get("/perplexity")
async def say_hello():
    return await PerplexityService().chat_request("I'm currently unemployed. How can the uk government assist me in finding me a job")

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
