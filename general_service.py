import os
from fastapi import HTTPException
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv

load_dotenv()


class GeneralRequest(BaseModel):
    prompt: str


def get_general_response(request: GeneralRequest):
    try:
        client = Groq(api_key=os.getenv('GROQ_API'))
        
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant that can discuss any topic and provide thoughtful responses."},
                {"role": "user", "content": request.prompt}
            ],
            temperature=0.7,
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