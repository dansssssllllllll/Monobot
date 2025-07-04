from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from transformers import pipeline, Conversation
import uvicorn
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

chatbot = pipeline("conversational", model="microsoft/DialoGPT-medium")
conversation = Conversation()

class ChatInput(BaseModel):
    message: str

@app.post("/chat")
def chat(input: ChatInput):
    conversation.add_user_input(input.message)
    result = chatbot(conversation)
    reply = result.generated_responses[-1]
    monobot_reply = f"💖 MonoBot: {reply.strip()} 😊"
    return {"reply": monobot_reply}

@app.get("/")
def get_index():
    return FileResponse("index.html")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
