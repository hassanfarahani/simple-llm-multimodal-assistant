from fastapi import APIRouter
from app.schemas import Message, MessageCreate
from fastapi import FastAPI, HTTPException
from typing import List
from app.clients import openai
import json
import base64

router = APIRouter(prefix="/message", tags=["messages"])

@router.post('/')
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

def get_ticket_price(destination_city):
    ticket_prices = {"london": "$799", "paris": "$899", "tokyo": "$1400", "berlin": "$499"}
    print(f"Tool called for city {destination_city}")
    price = ticket_prices.get(destination_city.lower(), "Unknown ticket price")
    return f"The price of a ticket to {destination_city} is {price}"

def talker(message):
    response = openai.audio.speech.create(
      model="gpt-4o-mini-tts",
      voice="onyx",    # Also, try replacing onyx with alloy or coral
      input=message
    )
    return response.content

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