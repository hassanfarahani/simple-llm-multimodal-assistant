from fastapi import APIRouter
from fastapi import FastAPI, File, Request, UploadFile, HTTPException
from fastapi.concurrency import run_in_threadpool
import tempfile
import os
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, TextStreamer, BitsAndBytesConfig

router = APIRouter(prefix="/audio", tags=["meetingminutes"])


@router.post('/upload')
async def upload_audio(request: Request,file: UploadFile = File(...)):
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    transcription = await run_in_threadpool(transcribeAudio, file, audio_bytes, request.app.state.whisper_pipe)
    report = await run_in_threadpool(reportSummaryUsingCpu, transcription, request.app.state.qwen) 
    return { 'summary': report }


def transcribeAudio(file, audio_bytes, whisper_pipe):
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
def reportSummaryUsingCpu(transcription, qwen):
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

    tokenizer, model = qwen
    

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
