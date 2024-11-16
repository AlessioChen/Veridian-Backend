from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from career_service import CareerAdviceRequest, get_career_advice

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}

# {"prompt": "What career should I pursue?"}
@app.post("/career-advice")
async def career_advice(request: CareerAdviceRequest):
    return StreamingResponse(
        get_career_advice(request),
        media_type="text/event-stream"
    )
