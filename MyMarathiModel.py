import os
import os.path
import torch
from transformers import pipeline

model_id = "SameerMahajan/whisper-tiny-retrained"

device = "cuda:0" if torch.cuda.is_available() else "cpu"

from transformers import WhisperTokenizer

tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-tiny", language="Marathi", task="transcribe")

pipe = pipeline(
   "automatic-speech-recognition",
   model=model_id,
   tokenizer=tokenizer,
   device=device,
)

for number in range(1,21,1):
    for i in range(50):
        audio_file = './samples/' + str(number) + '/' + str(number) + '_' + str(i) + '.wav'
        if os.path.isfile(audio_file):
            out = pipe(audio_file)
            print (audio_file, out)
