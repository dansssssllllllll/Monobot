from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from transformers import pipeline, Conversation

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Load the conversational model
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
