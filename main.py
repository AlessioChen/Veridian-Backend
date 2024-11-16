from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from llm_service import LLMService, ChatRequest
from pathlib import Path
from groq_services import GroqServices
import os
import json

app = FastAPI()
llm_service = LLMService()

# Configuration
MAX_FILE_SIZE = 25 * 1024 * 1024  # 25MB

parent_directory = os.path.dirname(os.getcwd())
AUDIO_DIR = f"{parent_directory}/audio-files"

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

    try:
        if not file:
            return JSONResponse(
                status_code=400,
                content={"error": "No file uploaded"}
            )

        file_path = SAVE_DIR / file.filename
        with file_path.open("wb") as audio_file:
            audio_file.write(await file.read())

        filename = os.path.dirname(__file__) + f"/{file_path}"
        transcription = GroqServices.speech_to_text(filename)
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
