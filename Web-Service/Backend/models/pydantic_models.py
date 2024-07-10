from pydantic import BaseModel
from typing import List, Dict

class ChatRequest(BaseModel):
    query: str
    conversation_string: str

class ChatResponse(BaseModel):
    response: str
    movies_desc: str