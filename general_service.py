import os
from fastapi import HTTPException
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain.chains import ConversationChain
from chat_memory import get_session_history
from typing import AsyncGenerator
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory


class GeneralRequest(BaseModel):
    prompt: str

async def get_general_response(request: GeneralRequest, user_id: str) -> AsyncGenerator[str, None]:
    try:
        llm = ChatGroq(
            groq_api_key=os.getenv('GROQ_API'),
            model_name="llama-3.1-8b-instant",
            temperature=0.7,
            max_tokens=1024,
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful AI assistant."),
            ("human", "{input}")
        ])
        
        chain = prompt | llm
        
        chain_with_history = RunnableWithMessageHistory(
            chain,
            get_session_history,
            input_messages_key="input",
        )

        async for event in chain_with_history.astream_events(
            {"input": request.prompt},
            config={"configurable": {"session_id": user_id}},
            version="v2"
        ):
            kind = event["event"]
            if kind == "on_chat_model_stream":
                content = event["data"]["chunk"].content
                yield content
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 