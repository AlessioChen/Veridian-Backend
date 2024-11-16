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

origins = ["http://localhost:3000", "http://127.0.0.1:8000"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/chat")
async def chat(request: ChatRequest):
    user_id = "default_user"
    
    async def generate_response():
        try:
            response_started = False
            async for token in llm_service.generate_response(user_id, request.message):
                if token:
                    print(f"API sending token: {token}")  # Debug log
                    message = {
                        "type": "message",
                        "content": token
                    }
                    encoded_message = f"data: {json.dumps(message)}\n\n".encode('utf-8')
                    print(f"Encoded message: {encoded_message}")  # Debug log
                    yield encoded_message
                    response_started = True
            
            if response_started:
                yield f"data: {json.dumps({'type': 'done'})}\n\n".encode('utf-8')
            
        except Exception as e:
            print(f"Error in API: {str(e)}")  # Debug log
            error_message = {
                "type": "error",
                "content": str(e)
            }
            yield f"data: {json.dumps(error_message)}\n\n".encode('utf-8')
    
    return StreamingResponse(
        generate_response(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream",
            "X-Accel-Buffering": "no"
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
