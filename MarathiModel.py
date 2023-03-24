import torch
from transformers import pipeline

model_id = "shripadbhat/whisper-tiny-mr"

device = "cuda:0" if torch.cuda.is_available() else "cpu"

pipe = pipeline(
   "automatic-speech-recognition",
   model=model_id,
   device=device,
)

for i in range(50):
    audio_file = './samples/1_' + str(i) + '.wav'
    out = pipe(audio_file)
    print (audio_file, out)
