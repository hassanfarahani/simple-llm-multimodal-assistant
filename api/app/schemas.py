
from pydantic import BaseModel, Field
from typing import List

class Message(BaseModel):
    role: str
    content: str

class MessageCreate(BaseModel):
    content: str
    history: List[Message] = Field(default_factory=list) 
