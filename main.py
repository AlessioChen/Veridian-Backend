from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse, StreamingResponse
import os
from datetime import datetime
from typing import Optional
from groq_services import GroqServices
from career_service import CareerAdviceRequest, get_career_advice

parent_directory = os.path.dirname(os.getcwd())
AUDIO_DIR = f"{parent_directory}/audio-files"

app = FastAPI(title="MP3 Processing API")

# Configuration
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


async def process_mp3(file_data: bytes, filename: str) -> str:
    return await GroqServices.process_audio_to_text(file_data, filename)


# Usage {"prompt": "What career should I pursue?"}
@app.post("/career-advice")
async def career_advice(request: CareerAdviceRequest):
    return StreamingResponse(
        get_career_advice(request),
        media_type="text/event-stream"
    )