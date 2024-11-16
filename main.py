from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from llm_service import LLMService, ChatRequest
import os
import json

app = FastAPI()
llm_service = LLMService()

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
