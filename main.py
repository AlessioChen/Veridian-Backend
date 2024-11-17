from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware


from llm_service import LLMService, ChatRequest
from groq_services import GroqServices
from perplexity_service import PerplexityService
from models.user_profile import UserProfile

from pathlib import Path
import os

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
    
    try:

        response = ""
        
        async for chunk in llm_service.generate_response("default_user", request.message):
            content = chunk.get('content', '')
            if content:
                response += content  
        
        return JSONResponse(
            status_code=200,
            content={
                "response": response,
            }
        )
    
    except Exception as e:
        print(f"Error in endpoint generate_response: {str(e)}")
        return "Sorry, something went wrong. Please try again later."




@app.post("/user-profile")
async def create_profile(user_profile: UserProfile):
    x = user_profile

@app.get("/perplexity")
async def say_hello():
    return await PerplexityService().chat_request("I'm currently unemployed. How can the uk government assist me in finding me a job")

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
