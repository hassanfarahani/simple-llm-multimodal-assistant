This is my first LLM AI Assistant app. My main goal for this very simple project is to get familiar with how LLm is working and how I can use tools in LLm to customize the frontier llm models for my own business app. 
These are the used technologies:
1. OpenAI aapi
2. vue3 for the frontend
3. fastapi for the backend

This app will call 4 LLM models (text, audio, image, etc )


Ok, I added another feature which is Creating meeting minutes from an Audio file. what it does is user upload a mp3 file and then, the new api endpoint (/audio/upload) will first transcribe the audio file uisng the model "openai/whisper-small.en" and then, it will summarize the transcript uisng the model "Qwen/Qwen2.5-0.5B-Instruct".

I downloaded some Denver City Council meeting minutes and selected a portion of the meeting for us to transcribe. I downloaded it here:  
https://drive.google.com/file/d/1N_kpSojRR5RYzupz6nqM8hMSoEF_R7pU/view?usp=sharing

original data: the HuggingFace dataset is [here](https://huggingface.co/datasets/huuuyeah/meetingbank) and the audio can be downloaded [here](https://huggingface.co/datasets/huuuyeah/MeetingBank_Audio/tree/main).
I learned how to use pipeline from Hugging Face Transformers for the open source models. 

