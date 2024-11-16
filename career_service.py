import os
from fastapi import HTTPException
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class CareerAdviceRequest(BaseModel):
    prompt: str

async def get_career_advice(request: CareerAdviceRequest):
    try:
        client = Groq(api_key=os.getenv('GROQ_API'))
        
        completion = client.chat.completions.create(
            model="llama-3.2-90b-text-preview",
            messages=[
                {"role": "system", "content": "You are a career advisor specializing in helping people make strategic career transitions to increase their earnings."},
                {"role": "user", "content": request.prompt}
            ],
            temperature=0,
            max_tokens=1024,
            top_p=1,
            stream=True,
            stop=None
        )
        
        for chunk in completion:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 