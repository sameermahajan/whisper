from transformers import pipeline
import gradio as gr

from transformers import WhisperTokenizer

tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-tiny", language="Marathi", task="transcribe")

pipe = pipeline(model="SameerMahajan/whisper-tiny-retrained",
       tokenizer=tokenizer,)  # change to "your-username/the-name-you-picked"

def transcribe(audio):
    text = pipe(audio)["text"]
    return text

iface = gr.Interface(
    fn=transcribe, 
    inputs=gr.Audio(source="microphone", type="filepath"), 
    outputs="text",
    title="Whisper Tiny Marathi",
    description="Realtime demo for Marathi speech recognition of numbers 1 through 20 using a fine-tuned Whisper tiny model.",
)

iface.launch()