import os
from typing import AsyncGenerator, Dict, Any, Annotated
from enum import Enum
from pydantic import BaseModel
from fastapi import HTTPException
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage, AIMessageChunk
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END, START
from typing import TypedDict, Sequence, Union, cast
from langgraph.graph.message import add_messages

class AgentType(str, Enum):
    CAREER = "career"
    GENERAL = "general"

class ChatRequest(BaseModel):
    message: str

class ChatState(TypedDict):
    messages: Annotated[list, add_messages]
    agent_type: str

class LLMService:
    def __init__(self):
        print("Initializing LLM Service")  # Debug
        # Initialize LLM configurations with streaming enabled
        self.router_llm = ChatGroq(
            groq_api_key=os.getenv('GROQ_API'),
            model_name="llama-3.1-8b-instant",
            temperature=0,
            max_tokens=256,
            streaming=True
        )
        
        self.agent_llm = ChatGroq(
            groq_api_key=os.getenv('GROQ_API'),
            model_name="llama-3.2-90b-text-preview",
            temperature=0.7,
            max_tokens=1024,
            streaming=True
        )
        
        # Initialize prompts
        self.router_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an intelligent router that determines which specialized agent should handle user requests.
            - Use 'career' for career advice, job searching, and professional development
            - Use 'general' for all other topics and general conversation
            
            Respond with only one word: either 'career' or 'general'"""),
            ("human", "{message}")
        ])
        
        self.agent_prompts = {
            AgentType.CAREER: ChatPromptTemplate.from_messages([
                ("system", "You are a career advisor specializing in helping people make strategic career transitions."),
                MessagesPlaceholder(variable_name="messages"),
                ("human", "{message}")
            ]),
            AgentType.GENERAL: ChatPromptTemplate.from_messages([
                ("system", "You are a helpful AI assistant."),
                MessagesPlaceholder(variable_name="messages"),
                ("human", "{message}")
            ])
        }
        
        # Initialize the graph
        self.workflow = self._create_graph()

    async def route_message(self, state: ChatState) -> ChatState:
        print("Routing message")  # Debug
        messages = state["messages"]
        last_message = cast(HumanMessage, messages[-1])
        
        chain = self.router_prompt | self.router_llm
        result = await chain.ainvoke({"message": last_message.content})
        print(f"Routed to: {result.content}")  # Debug
        
        try:
            agent_type = AgentType(result.content.strip().lower())
            state["agent_type"] = agent_type
        except ValueError:
            state["agent_type"] = AgentType.GENERAL
            
        return state

    async def generate_agent_response(self, state: ChatState) -> AsyncGenerator[ChatState, None]:
        print("Generating agent response")  # Debug
        messages = state["messages"]
        last_message = cast(HumanMessage, messages[-1])
        agent_type = state["agent_type"]
        
        prompt = self.agent_prompts[agent_type]
        chain = prompt | self.agent_llm
        
        # Use astream instead of ainvoke
        async for chunk in chain.astream({
            "message": last_message.content,
            "messages": messages[:-1]
        }):
            print(f"Streaming chunk: {chunk.content}")  # Debug
            if chunk.content:
                # Create a new state for each chunk
                new_state = ChatState(
                    messages=state["messages"],
                    agent_type=state["agent_type"]
                )
                new_state["messages"].append(AIMessage(content=chunk.content))
                yield new_state

    def _create_graph(self) -> StateGraph:
        workflow = StateGraph(ChatState)
        
        workflow.add_node("route", self.route_message)
        workflow.add_node("generate", self.generate_agent_response)
        
        workflow.add_edge(START, "route")
        workflow.add_edge("route", "generate")
        workflow.add_edge("generate", END)
        
        return workflow.compile()

    async def generate_response(self, user_id: str, message: str) -> AsyncGenerator[str, None]:
        try:
            print(f"Starting response generation for message: {message}")  # Debug
            state = ChatState(
                messages=[HumanMessage(content=message)],
                agent_type=""
            )
            
            previous_content = ""
            async for step in self.workflow.astream(state, stream_mode="node"):  # Add stream_mode="node"
                print(f"Received step: {step}")  # Debug
                if isinstance(step, dict) and "messages" in step:
                    messages = step["messages"]
                    if messages and isinstance(messages[-1], AIMessage):
                        current_content = messages[-1].content
                        if current_content != previous_content:
                            new_content = current_content[len(previous_content):]
                            print(f"Yielding new content: {new_content}")  # Debug
                            yield new_content
                            previous_content = current_content
                    
        except Exception as e:
            print(f"Error in generate_response: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e)) 