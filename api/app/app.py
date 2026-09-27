import os
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from app.schemas import Message
from app.schemas import MessageCreate
from openai import OpenAI
import json
from typing import List
import base64
from io import BytesIO
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException

import tempfile
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, TextStreamer, BitsAndBytesConfig
import torch
from huggingface_hub import login
from fastapi.concurrency import run_in_threadpool
from contextlib import asynccontextmanager
from functools import lru_cache

load_dotenv()

HF_TOKEN = os.environ.get("HF_TOKEN")
if HF_TOKEN:
    login(token=HF_TOKEN, add_to_git_credential=False)


whisper_pipe = None


@lru_cache(maxsize=1)
def get_qwen():
    """Load Qwen lazily and cache it. First call is slow, rest are instant."""
    QWEN = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(QWEN)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(QWEN, torch_dtype=torch.float32)
    model.eval()
    return tokenizer, model


@asynccontextmanager
async def lifespan(app: FastAPI):
    global whisper_pipe

    # Whisper: load eagerly at startup so the first request isn't slow.
    whisper_pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-small.en",
        dtype=torch.float16,
        device="cpu",
        return_timestamps=True,
        chunk_length_s=30,
        stride_length_s=5,
    )

    # Qwen: warm it too, so the first report request is fast.
    get_qwen()

    yield

    # Optional cleanup
    whisper_pipe = None



app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

openai_api_key = os.getenv('OPENAI_API_KEY')
if openai_api_key:
    print(f"****** OpenAI API Key exists and begins {openai_api_key[:8]} ****")
else:
    print("OpenAI API Key not set")
    
openai = OpenAI()


@app.post('/message')
def get_message(msg: MessageCreate):
    try:
        res = chat(msg.content, msg.history)
        return res
    except Exception as e:
        print(f'Error: {e}')
        raise HTTPException(status_code=500, detail=str(e))

def chat(message: str, history: List[Message]):
    MODEL = "gpt-4.1-mini"
    system_message = """
    You are a helpful assistant for an Airline called FlightAI.
    Give short, courteous answers, no more than 1 sentence.
    Always be accurate. If you don't know the answer, say so.
    """

    price_function = {
        "name": "get_ticket_price",
        "description": "Get the price of a return ticket to the destination city.",
        "parameters": {
            "type": "object",
            "properties": {
                "destination_city": {
                    "type": "string",
                    "description": "The city that the customer wants to travel to",
                },
            },
            "required": ["destination_city"],
            "additionalProperties": False
        }
    }
    tools = [{"type": "function", "function": price_function}]
    try:
        history = [{"role":h.role, "content":h.content} for h in history]
        messages = [{"role": "system", "content": system_message}] + history + [{"role": "user", "content": message}]
        response = openai.chat.completions.create(model=MODEL, messages=messages, tools=tools)
        cities = []
        image_data_uri = None
        while response.choices[0].finish_reason=="tool_calls":
            message = response.choices[0].message
            responses, cities = handle_tool_calls_and_return_cities(message)
            messages.append(message)
            messages.extend(responses)
            response = openai.chat.completions.create(model=MODEL, messages=messages, tools=tools)

        reply = response.choices[0].message.content
        history += [{"role":"assistant", "content":reply}]
        voice_bytes = talker(reply)        
        # Convert bytes to Base64 string for JSON transport
        voice_base64 = base64.b64encode(voice_bytes).decode('utf-8')      

        if cities:
            image_data_uri = generate_image(cities[0])

        return {
            "model_response": reply,
            "audio_base64": voice_base64,
            "destination_city_image": image_data_uri 
        }
    except Exception as e:
        print(f'OpenAI API Error: {e}')
        raise

def handle_tool_calls(message):
    responses = []
    for tool_call in message.tool_calls:
        if tool_call.function.name == "get_ticket_price":
            arguments = json.loads(tool_call.function.arguments)
            city = arguments.get('destination_city')
            price_details = get_ticket_price(city)
            responses.append({
                "role": "tool",
                "content": price_details,
                "tool_call_id": tool_call.id
            })
    return responses

def handle_tool_calls_and_return_cities(message):
    responses = []
    cities = []
    for tool_call in message.tool_calls:
        if tool_call.function.name == "get_ticket_price":
            arguments = json.loads(tool_call.function.arguments)
            city = arguments.get('destination_city')
            cities.append(city)
            price_details = get_ticket_price(city)
            responses.append({
                "role": "tool",
                "content": price_details,
                "tool_call_id": tool_call.id
            })
    return responses, cities


def generate_image(city):
    image_response = openai.images.generate(
            model="gpt-image-1-mini",
            prompt=f"An image representing a vacation in {city}, showing tourist spots and everything unique about {city}, in a vibrant pop-art style",
            size="1024x1024",
            n=1,
        )
    image_base64 = image_response.data[0].b64_json
    return f"data:image/png;base64,{image_base64}"

def talker(message):
    response = openai.audio.speech.create(
      model="gpt-4o-mini-tts",
      voice="onyx",    # Also, try replacing onyx with alloy or coral
      input=message
    )
    return response.content

def get_ticket_price(destination_city):
    ticket_prices = {"london": "$799", "paris": "$899", "tokyo": "$1400", "berlin": "$499"}
    print(f"Tool called for city {destination_city}")
    price = ticket_prices.get(destination_city.lower(), "Unknown ticket price")
    return f"The price of a ticket to {destination_city} is {price}"

# ***********************************************************************************************
# Option 1: Use Open Source for Transcription - Hugging Face Pipelines
# the openai/whisper-small.en model will be stored in opt/huggingface/hub inside the container
# the audio file will be stored in /tmp inside the container (/tmp/tmpj8ph40a9.mp3)
@app.post('/audio/upload')
async def upload_audio(file: UploadFile = File(...)):
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    transcription = await run_in_threadpool(transcribeAudio, file, audio_bytes)
    report = await run_in_threadpool(reportSummaryUsingCpu, transcription) 
    return { 'summary': report }


def transcribeAudio(file, audio_bytes):
   # Whisper's pipeline needs a file path or a numpy array — NOT a file object.
    # Simplest approach: write to a temp file and let transformers/ffmpeg read it.
    suffix = ".mp3"
    if file.filename and "." in file.filename:
        suffix = "." + file.filename.rsplit(".", 1)[-1]

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    
    try:
        result = whisper_pipe(tmp_path)
        return result["text"]
        # return {"transcription": result["text"]}
    finally:
        os.unlink(tmp_path)

# this function needs GPU to run
def reportSummary(transcription):
    # LLAMA = "unsloth/Llama-3.2-1B-Instruct-GGUF"
    # LLAMA = "meta-llama/Llama-3.2-3B-Instruct"
    QWEN = "Qwen/Qwen2.5-0.5B-Instruct"
    system_message = """
    You produce minutes of meetings from transcripts, with summary, key discussion points,
    takeaways and action items with owners, in markdown format without code blocks.
    """

    user_prompt = f"""
    Below is an extract transcript of a Denver council meeting.
    Please write minutes in markdown without code blocks, including:
    - a summary with attendees, location and date
    - discussion points
    - takeaways
    - action items with owners

    Transcription:
    {transcription}
    """

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_prompt}
    ]

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4"
    )

    tokenizer = AutoTokenizer.from_pretrained(QWEN)
    tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")
    streamer = TextStreamer(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(QWEN, device_map="auto", quantization_config=quant_config)
    outputs = model.generate(inputs, max_new_tokens=2000, streamer=streamer)
    response = tokenizer.decode(outputs[0])
    return response

# on CPU, we cannot do quantization at all
def reportSummaryUsingCpu(transcription):
    QWEN = "Qwen/Qwen2.5-0.5B-Instruct"
    system_message = """
    You produce minutes of meetings from transcripts, with summary, key discussion points,
    takeaways and action items with owners, in markdown format without code blocks.
    """

    user_prompt = f"""
    Below is an extract transcript of a Denver council meeting.
    Please write minutes in markdown without code blocks, including:
    - a summary with attendees, location and date
    - discussion points
    - takeaways
    - action items with owners

    Transcription:
    {transcription}
    """

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_prompt}
    ]

    tokenizer, model = get_qwen()

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
    )

    # Move tensors to the model's device (CPU here) — no-op if already there.
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    prompt_len = inputs["input_ids"].shape[-1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=2000,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = outputs[0][prompt_len:]
    return tokenizer.decode(generated, skip_special_tokens=True)


