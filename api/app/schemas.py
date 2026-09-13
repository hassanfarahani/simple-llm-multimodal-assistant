# from pydantic import BaseModel
# from fastapi_users import schemas

# from typing import List

# class Message(BaseModel):
#     role: str  # "user" or "assistant"
#     content: str

# class MessageCreate(BaseModel):
#     content: str
#     history: List[Message] = [] 

from pydantic import BaseModel, Field
from typing import List

class Message(BaseModel):
    role: str
    content: str

class MessageCreate(BaseModel):
    content: str
    history: List[Message] = Field(default_factory=list)  # ✅ Fixed