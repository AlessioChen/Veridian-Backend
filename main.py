from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse, StreamingResponse
import os
import aiohttp
import tempfile
from datetime import datetime
from typing import Optional
from career_service import CareerAdviceRequest, get_career_advice
from general_service import GeneralRequest, get_general_response
from agent_router import Router, RouterRequest
from pydantic import BaseModel

parent_directory = os.path.dirname(os.getcwd())
AUDIO_DIR = f"{parent_directory}/audio-files"

app = FastAPI(title="MP3 Processing API")
router = Router()

# CORS configuration
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Configure appropriately for production
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
MAX_FILE_SIZE = 25 * 1024 * 1024  # 25MB


@app.post("/transcribe/mp3")
async def transcribe_mp3(
        mp3_file: UploadFile = File(..., description="MP3 audio file to transcribe"),
        language: Optional[str] = Form(None, description="Optional language code (e.g., 'en', 'es')")
):
    """
    Endpoint to transcribe MP3 audio data

    Parameters:
    - mp3_file: MP3 file sent as form-data
    - language: Optional language code for transcription

    Returns:
    - JSON with transcription results
    """
    try:
        # Validate the MP3 file
        await validate_mp3(mp3_file)

        # Read the file content
        file_content = await mp3_file.read()

        # Process the MP3 data
        result = await process_mp3(file_content, mp3_file.filename)

        return JSONResponse(
            content={
                "success": True,
                "filename": mp3_file.filename,
                "timestamp": datetime.now().isoformat(),
                "file_size_bytes": len(file_content),
                "transcription": result,
                "language": language
            },
            status_code=200
        )

    except HTTPException as http_ex:
        raise http_ex
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


# Test endpoint
@app.get("/test")
async def test_endpoint():
    """Test endpoint to verify API is running"""
    return {"status": "ok", "message": "MP3 processing endpoint is active"}


async def validate_mp3(file: UploadFile) -> None:
    """
    Validate the uploaded MP3 file
    """
    # Check content type
    if file.content_type not in ['audio/mpeg', 'audio/mp3']:
        raise HTTPException(
            status_code=415,
            detail="File must be MP3 format"
        )

    # Check file size
    file_size = 0
    chunk_size = 8192  # Read in 8KB chunks

    while chunk := await file.read(chunk_size):
        file_size += len(chunk)
        if file_size > MAX_FILE_SIZE:
            await file.seek(0)  # Reset file pointer
            raise HTTPException(
                status_code=413,
                detail=f"File size exceeds maximum limit of {MAX_FILE_SIZE / 1024 / 1024}MB"
            )

    # Reset file pointer for subsequent reads
    await file.seek(0)


async def process_mp3(file_data: bytes, filename: str) -> dict:
    """
    Process MP3 data with Groq's Whisper API
    """
    # Create temporary file
    with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
        temp_file.write(file_data)
        temp_path = temp_file.name

    try:
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "multipart/form-data"
        }

        form_data = aiohttp.FormData()
        form_data.add_field(
            'file',
            open(temp_path, 'rb'),
            filename=filename,
            content_type='audio/mpeg'
        )
        form_data.add_field('model', 'whisper-1')

        async with aiohttp.ClientSession() as session:
            async with session.post(
                    GROQ_API_URL,
                    headers=headers,
                    data=form_data
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    raise HTTPException(
                        status_code=response.status,
                        detail=f"Transcription API error: {error_text}"
                    )
    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.unlink(temp_path)
@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}

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
